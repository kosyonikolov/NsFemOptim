#ifndef LIBS_CUDAUTILS_INCLUDE_CU_GAUSSSEIDEL
#define LIBS_CUDAUTILS_INCLUDE_CU_GAUSSSEIDEL

#include <memory>

#include <linalg/csrMatrix.h>

#include <cu/blas.h>
#include <cu/csrF.h>
#include <cu/spmv.h>
#include <cu/spmm.h>

namespace cu
{
    class GaussSeidel
    {
        cu::Blas & blas;

        std::unique_ptr<csrF> m; // reordered

        std::unique_ptr<spmv> mSpmv; // Used for single-channel
        std::unique_ptr<spmm> mSpmm; // For two-channel

        // Stripped matrix + inverted diagonal
        cu::vec<float> values;
        cu::vec<float> invDiag;
        cu::vec<int> column;
        cu::vec<int> rowStart;

        std::vector<int> cpuColoring;
        cu::vec<int> coloring;

        std::vector<int> partitionStart; // size = numColors + 1, last element is a sentinel (== coloring.size())

        // Internal reordered vectors
        cu::vec<float> rhs;
        cu::vec<float> sol;

        // Block & grid sizes for the kernels
        dim3 reorderBlockSize, reorderGridSize;
        std::vector<dim3> blockSize;
        std::vector<dim3> gridSize;

        int numCh;

        // Check MSE if iter % mseMod == 0
        int mseMod = 1;

        float solve1(const int maxIters, const float target);

        float solve2(const int maxIters, const float target);

    public:
        // Input/output vectors
        // Do not reallocate!
        cu::vec<float> ioRhs;
        cu::vec<float> ioSol;

        // Output statistics
        std::vector<float> lastMse;
        int lastIterations = -1;

        GaussSeidel(cu::Blas & blas, cusparseHandle_t sparseHandle, const linalg::CsrMatrix<float> & cpuMatrix,
                    const int numCh = 1);

        // -> MSE of solution
        float solve(const int maxIters, const float target);

        void setMseCheckInterval(const int newInterval);
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_GAUSSSEIDEL */
