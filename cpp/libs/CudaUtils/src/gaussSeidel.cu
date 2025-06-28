#include <cu/gaussSeidel.h>

#include <algorithm>
#include <cassert>
#include <stdexcept>

#include <linalg/gaussSeidel.h>
#include <linalg/graphs.h>

#include <utils/stopwatch.h>

#include <cu/stopwatch.h>

namespace cu
{
    // Perform a step of the Gauss-Seidel algorithm on a partition of the system Mx = b
    // M is a square CSR matrix described by values, column and rowStart
    // The rows which are updated are given by (partition, partitionSize)
    __global__ void gaussSeidelStepPartition(float * x, const float * b,
                                             const float * values, const int * column, const int * rowStart,
                                             const int * partition, const int partitionSize)
    {
        int i0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int i = i0; i < partitionSize; i += stride)
        {
            const int row = partition[i];
            const int j1 = rowStart[row + 1];
            float diag = 0; // element at (i, i)
            float negSum = 0;
            for (int j = rowStart[row]; j < j1; j++)
            {
                const int col = column[j];
                // TODO Version where the inverted diagonal is given at the input and the matrix doesn't have diagonal entries - faster?
                if (col == row)
                {
                    diag = values[j];
                }
                else
                {
                    negSum += values[j] * x[col];
                }
            }
            x[row] = (b[row] - negSum) / diag;
        }
    }

    // Perform a step of the Gauss-Seidel algorithm on a partition of the system Mx = b
    // M is a square CSR matrix with  described by values, column and rowStart
    // The sparse structure contains NO diagonal elements
    // Insteady their multiplicative inverses are in invDiag
    // The rows which are updated are given by (partition, partitionSize)
    __global__ void gaussSeidelStepPartitionInvDiag(float * x, const float * b, const float * invDiag,
                                                    const float * values, const int * column, const int * rowStart,
                                                    const int * partition, const int partitionSize)
    {
        int i0 = blockIdx.x * blockDim.x + threadIdx.x;
        const int stride = blockDim.x * gridDim.x;
        for (int i = i0; i < partitionSize; i += stride)
        {
            const int row = partition[i];
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

    GaussSeidel::GaussSeidel(cu::Blas & blas, cusparseHandle_t sparseHandle, const linalg::CsrMatrix<float> & cpuMatrix)
        : blas(blas), m(cpuMatrix), mSpmv(sparseHandle, m),
          coloring(cpuMatrix.cols), rhs(cpuMatrix.cols), sol(cpuMatrix.cols)
    {
        assert(cpuMatrix.cols == cpuMatrix.rows);
        const int n = cpuMatrix.cols;

        // Create a stripped matrix (no diagonal) and the inverted diagonal
        auto ctx = linalg::buildGaussSeidelContext(cpuMatrix);
        invDiag.overwriteUpload(ctx.invDiag);
        values.overwriteUpload(ctx.stripped.values);
        column.overwriteUpload(ctx.stripped.column);
        rowStart.overwriteUpload(ctx.stripped.rowStart);

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

        // Upload the coloring
        coloring.upload(cpuColoring);
    }

    float GaussSeidel::solve(const int maxIters, const float target)
    {
        float lastMse = -1;

        // Calculate block and grid sizes for each partition
        const int nParts = partitionStart.size() - 1;
        std::vector<dim3> blockSize(nParts);
        std::vector<dim3> gridSize(nParts);
        constexpr int maxThreads = 512;
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

        Stopwatch sw;
        u::Stopwatch bigSw;
        for (int iter = 0; iter < maxIters; iter++)
        {
            bigSw.reset();
            sw.reset();

            // Perform the updates
            for (int p = 0; p < nParts; p++)
            {
                const int j0 = partitionStart[p];
                const int j1 = partitionStart[p + 1];
                const int pSize = j1 - j0;

                // Send it
                const dim3 currGrid = gridSize[p];
                const dim3 currBlock = blockSize[p];
                // gaussSeidelStepPartition<<<currGrid, currBlock>>>(sol.get(), rhs.get(),
                //                                                   m.values.get(), m.column.get(), m.rowStart.get(),
                //                                                   coloring.get() + j0, pSize);
                gaussSeidelStepPartitionInvDiag<<<currGrid, currBlock>>>(sol.get(), rhs.get(), invDiag.get(),
                                                                         values.get(), column.get(), rowStart.get(),
                                                                         coloring.get() + j0, pSize);
            }

            const auto tGs = sw.millis(true);

            // auto rc = cudaStreamSynchronize(0);

            // Calculate MSE
            auto & res = mSpmv.b;
            mSpmv.compute(sol, res);
            const int n = sol.size();
            cu::saxpy(blas, n, rhs.get(), res.get(), -1.0f);

            float norm2 = -1; // == sqrt(sum(res[i]^2))
            auto rc = cublasSnrm2(blas.handle, n, res.get(), 1, &norm2);
            if (rc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            {
                throw std::runtime_error(std::format("cublasSnrm2 failed: {}", cublasGetStatusName(rc)));
            }

            const auto tMse = sw.millis();

            const float mse = norm2 / std::sqrt(n);
            lastMse = mse;

            const auto tIter = bigSw.millis();
            std::cout << iter << ": " << mse << " (gs = " << tGs << " ms, mse = " << tMse << " ms, total = " << tIter << " ms)\n";
            if (mse < target)
            {
                break;
            }
        }

        return lastMse;
    }
} // namespace cu
