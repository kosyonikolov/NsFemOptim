#include <cu/gaussSeidel.h>

#include <algorithm>
#include <array>
#include <cassert>
#include <stdexcept>

#include <linalg/gaussSeidel.h>
#include <linalg/graphs.h>

#include <utils/stopwatch.h>

#include <cu/stopwatch.h>

namespace cu
{
    __global__ void reorderXbFwd(const float * srcX, const float * srcB,
                                 float * dstX, float * dstB,
                                 const int * coloring, const int n)
    {
        int i0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int i = i0; i < n; i += stride)
        {
            const int j = coloring[i];
            dstX[i] = srcX[j];
            dstB[i] = srcB[j];
        }
    }

    __global__ void reorderXbFwd2ch(const float * srcX, const float * srcB,
                                    float * dstX, float * dstB,
                                    const int * coloring, const int n)
    {
        int i0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int i = i0; i < n; i += stride)
        {
            const int j = coloring[i];
            dstX[i] = srcX[j];
            dstB[i] = srcB[j];
            dstX[i + n] = srcX[j + n];
            dstB[i + n] = srcB[j + n];
        }
    }

    __global__ void reorderXInv(const float * srcX, float * dstX,
                                const int * coloring, const int n)
    {
        int i0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int i = i0; i < n; i += stride)
        {
            const int j = coloring[i];
            dstX[j] = srcX[i];
        }
    }

    __global__ void reorderXInv2ch(const float * srcX, float * dstX,
                                   const int * coloring, const int n)
    {
        int i0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int i = i0; i < n; i += stride)
        {
            const int j = coloring[i];
            dstX[j] = srcX[i];
            dstX[j + n] = srcX[i + n];
        }
    }

    // Perform a step of the Gauss-Seidel algorithm on a partition of the system Mx = b
    // The partition covers rows [partitionStart, partitionEnd)
    // The matrix is in CSR format, described by values, column and rowStart
    // It has no diagonal entries - the multiplicative inverse of the original matrix are stored in invDiag instead
    __global__ void gaussSeidelStepPartitionInvDiagR(float * x, const float * b, const float * invDiag,
                                                     const float * values, const int * column, const int * rowStart,
                                                     const int partitionStart, const int partitionEnd)
    {
        int row0 = partitionStart + blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int row = row0; row < partitionEnd; row += stride)
        {
            const int j1 = rowStart[row + 1];
            float negSum = 0;
            for (int j = rowStart[row]; j < j1; j++)
            {
                const int col = column[j];
                // col is never equal to row
                negSum += values[j] * x[col];
            }
            x[row] = (b[row] - negSum) * invDiag[row];
        }
    }

    __global__ void gaussSeidelStepPartitionInvDiagR2ch(float * x, const float * b, const float * invDiag,
                                                        const float * values, const int * column, const int * rowStart,
                                                        const int partitionStart, const int partitionEnd,
                                                        const int n)
    {
        int row0 = partitionStart + blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int row = row0; row < partitionEnd; row += stride)
        {
            const int j1 = rowStart[row + 1];
            float negSum0 = 0;
            float negSum1 = 0;
            for (int j = rowStart[row]; j < j1; j++)
            {
                const int col = column[j];
                // col is never equal to row
                negSum0 += values[j] * x[col];
                negSum1 += values[j] * x[col + n];
            }
            x[row] = (b[row] - negSum0) * invDiag[row];
            const int secondRow = row + n;
            x[secondRow] = (b[secondRow] - negSum1) * invDiag[row];
        }
    }

    GaussSeidel::GaussSeidel(cu::Blas & blas, cusparseHandle_t sparseHandle, const linalg::CsrMatrix<float> & cpuMatrix,
                             const int numCh)
        : blas(blas),
          coloring(cpuMatrix.cols),
          rhs(cpuMatrix.cols * numCh), sol(cpuMatrix.cols * numCh),
          numCh(numCh),
          ioRhs(cpuMatrix.cols * numCh), ioSol(cpuMatrix.cols * numCh)
    {
        assert(cpuMatrix.cols == cpuMatrix.rows);
        const int n = cpuMatrix.cols;

        if (numCh < 1 || numCh > 2)
        {
            throw std::invalid_argument("Only 1 and 2 ch Gauss-Seidel is supported");
        }

        // TODO Extract the matrix preprocessing to separate function?
        // Create a coloring of the matrix
        // Use the smallest-last ordering for now - it seems to produce good results
        auto graph = linalg::buildCsrGraph(cpuMatrix);
        assert(graph.size() == n);
        auto slOrder = linalg::buildSmallestLastOrdering(graph);
        auto parts = linalg::partitionGraphGreedy(graph, slOrder);

        const int nColors = parts.size();

        // Sort the individual partitions and place them in the coloring vector
        cpuColoring.resize(n);
        partitionStart.resize(nColors + 1);
        partitionStart.back() = n;
        int i = 0;
        for (int c = 0; c < nColors; c++)
        {
            auto & p = parts[c];
            std::sort(p.begin(), p.end());
            std::copy_n(p.begin(), p.size(), cpuColoring.begin() + i);
            partitionStart[c] = i;
            i += p.size();
        }

        // Reorder the matrix to make the coloring redundant -
        // first partition is [0, P1), second is [P1, P2) and so on
        auto reordered = cpuMatrix.slice(cpuColoring, cpuColoring);

        // Upload the reordered matrix
        m = std::make_unique<csrF>(reordered);

        if (numCh == 1)
        {
            mSpmv = std::make_unique<spmv>(sparseHandle, *m);
        }
        else
        {
            mSpmm = std::make_unique<spmm>(sparseHandle, *m, numCh);
        }

        // Create a stripped matrix (no diagonal) and the inverted diagonal
        auto ctx = linalg::buildGaussSeidelContext(reordered);
        invDiag.overwriteUpload(ctx.invDiag);
        values.overwriteUpload(ctx.stripped.values);
        column.overwriteUpload(ctx.stripped.column);
        rowStart.overwriteUpload(ctx.stripped.rowStart);

        // Upload the coloring
        coloring.upload(cpuColoring);

        // Calculate kernel sizes
        constexpr int maxThreads = 512;
        if (n <= maxThreads)
        {
            reorderBlockSize = dim3(n);
            reorderGridSize = dim3(1);
        }
        else
        {
            const int nB = (n + maxThreads - 1) / maxThreads;
            reorderBlockSize = dim3(maxThreads);
            reorderGridSize = dim3(nB);
        }

        // Calculate block and grid sizes for each partition
        const int nParts = partitionStart.size() - 1;
        blockSize.resize(nParts);
        gridSize.resize(nParts);
        for (int p = 0; p < nParts; p++)
        {
            const int pSize = partitionStart[p + 1] - partitionStart[p];
            if (pSize <= maxThreads)
            {
                blockSize[p] = dim3(pSize);
                gridSize[p] = dim3(1);
            }
            else
            {
                blockSize[p] = dim3(maxThreads);
                const int nBlocks = (pSize + maxThreads - 1) / maxThreads;
                gridSize[p] = dim3(nBlocks);
            }
        }

        lastMse.resize(numCh);
    }

    float GaussSeidel::solve1(const int maxIters, const float target)
    {
        const int n = coloring.size();
        const int nParts = partitionStart.size() - 1;

        // Reorder the IO vectors
        reorderXbFwd<<<reorderGridSize, reorderBlockSize>>>(ioSol.get(), ioRhs.get(),
                                                            sol.get(), rhs.get(),
                                                            coloring.get(), n);

        float lastMse = -1;

#ifdef CU_SOLVERS_ITER_LOG
        Stopwatch sw;
        u::Stopwatch bigSw;
#endif

        int iter = 0;
        for (; iter < maxIters; iter++)
        {
#ifdef CU_SOLVERS_ITER_LOG
            bigSw.reset();
            sw.reset();
#endif

            // Perform the updates
            for (int p = 0; p < nParts; p++)
            {
                const int j0 = partitionStart[p];
                const int j1 = partitionStart[p + 1];

                // Send it
                const dim3 currGrid = gridSize[p];
                const dim3 currBlock = blockSize[p];
                gaussSeidelStepPartitionInvDiagR<<<currGrid, currBlock>>>(sol.get(), rhs.get(), invDiag.get(),
                                                                          values.get(), column.get(), rowStart.get(),
                                                                          j0, j1);
            }

#ifdef CU_SOLVERS_ITER_LOG
            const auto tGs = sw.millis(true);
#endif

            bool done = false;
            float mse = lastMse;
            if (iter % mseMod == 0)
            {
                // Calculate MSE
                auto & res = mSpmv->b;
                mSpmv->compute(sol, res);
                const int n = sol.size();
                cu::saxpy(blas, n, rhs.get(), res.get(), -1.0f);

                float norm2 = -1; // == sqrt(sum(res[i]^2))
                auto rc = cublasSnrm2(blas.handle, n, res.get(), 1, &norm2);
                if (rc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                {
                    throw std::runtime_error(std::format("cublasSnrm2 failed: {}", cublasGetStatusName(rc)));
                }

                mse = norm2 / std::sqrt(n);
                lastMse = mse;
                if (mse < target)
                {
                    done = true;
                }
            }

#ifdef CU_SOLVERS_ITER_LOG
            const auto tMse = sw.millis();
            const auto tIter = bigSw.millis();
            std::cout << iter << ": " << mse << " (gs = " << tGs << " ms, mse = " << tMse << " ms, total = " << tIter << " ms)\n";
#endif
            if (done)
            {
                break;
            }
        }

        // Place the result in the IO vector
        reorderXInv<<<reorderGridSize, reorderBlockSize>>>(sol.get(), ioSol.get(), coloring.get(), n);

        this->lastMse[0] = lastMse;
        this->lastIterations = iter;

        return lastMse;
    }

    float GaussSeidel::solve2(const int maxIters, const float target)
    {
        const int n = coloring.size();
        const int nParts = partitionStart.size() - 1;

        // Reorder the IO vectors
        reorderXbFwd2ch<<<reorderGridSize, reorderBlockSize>>>(ioSol.get(), ioRhs.get(),
                                                               sol.get(), rhs.get(),
                                                               coloring.get(), n);
        std::array<float, 2> lastMse = {-1, -1};

#ifdef CU_SOLVERS_ITER_LOG
        Stopwatch sw;
        u::Stopwatch bigSw;
#endif

        int iter = 0;
        for (; iter < maxIters; iter++)
        {
#ifdef CU_SOLVERS_ITER_LOG
            bigSw.reset();
            sw.reset();
#endif

            // Perform the updates
            for (int p = 0; p < nParts; p++)
            {
                const int j0 = partitionStart[p];
                const int j1 = partitionStart[p + 1];

                // Send it
                const dim3 currGrid = gridSize[p];
                const dim3 currBlock = blockSize[p];
                gaussSeidelStepPartitionInvDiagR2ch<<<currGrid, currBlock>>>(sol.get(), rhs.get(), invDiag.get(),
                                                                             values.get(), column.get(), rowStart.get(),
                                                                             j0, j1, n);
            }

#ifdef CU_SOLVERS_ITER_LOG
            const auto tGs = sw.millis(true);
#endif

            bool done = false;
            std::array<float, 2> mse = {lastMse[0], lastMse[1]};
            if (iter % mseMod == 0)
            {
                // Calculate MSE
                auto & resXy = mSpmm->b;
                mSpmm->compute(sol, resXy);
                cu::saxpy(blas, 2 * n, rhs.get(), resXy.get(), -1.0f);

                for (int c = 0; c < 2; c++)
                {
                    float norm2 = -1; // == sqrt(sum(res[i]^2))
                    auto rc = cublasSnrm2(blas.handle, n, resXy.get() + c * n, 1, &norm2);
                    if (rc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                    {
                        throw std::runtime_error(std::format("cublasSnrm2 failed: {}", cublasGetStatusName(rc)));
                    }

                    mse[c] = norm2 / std::sqrt(n);
                    lastMse[c] = mse[c];
                }

                if (mse[0] < target && mse[1] < target)
                {
                    done = true;
                }
            }

#ifdef CU_SOLVERS_ITER_LOG
            const auto tMse = sw.millis();
            const auto tIter = bigSw.millis();
            std::cout << iter << ": " << mse[0] << " / " << mse[1] << " (gs = " << tGs << " ms, mse = " << tMse << " ms, total = " << tIter << " ms)\n";
#endif
            if (done)
            {
                break;
            }
        }

        // Place the result in the IO vector
        reorderXInv2ch<<<reorderGridSize, reorderBlockSize>>>(sol.get(), ioSol.get(), coloring.get(), n);

        // Average the channel MSE
        const float avgMse = std::sqrt(lastMse[0] * lastMse[0] + lastMse[1] * lastMse[1]);

        this->lastMse[0] = lastMse[0];
        this->lastMse[1] = lastMse[1];
        this->lastIterations = iter;

        return avgMse;
    }

    void GaussSeidel::setMseCheckInterval(const int newInterval)
    {
        if (newInterval < 1)
        {
            throw std::invalid_argument("MSE check interval should be at least 1");
        }

        mseMod = newInterval;
    }

    float GaussSeidel::solve(const int maxIters, const float target)
    {
        if (numCh == 1)
        {
            return solve1(maxIters, target);
        }
        else
        {
            assert(numCh == 2);
            return solve2(maxIters, target);
        }
    }
} // namespace cu
