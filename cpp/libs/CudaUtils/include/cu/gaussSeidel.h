#ifndef LIBS_CUDAUTILS_INCLUDE_CU_GAUSSSEIDEL
#define LIBS_CUDAUTILS_INCLUDE_CU_GAUSSSEIDEL

#include <memory>

#include <linalg/csrMatrix.h>

#include <cu/csrF.h>
#include <cu/spmv.h>
#include <cu/blas.h>

namespace cu
{
    class GaussSeidel
    {
        cu::Blas & blas;

        std::unique_ptr<csrF> m; // reordered
        std::unique_ptr<spmv> mSpmv;

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

    public:
        // Input/output vectors
        // Do not reallocate!
        cu::vec<float> ioRhs;        
        cu::vec<float> ioSol;

        GaussSeidel(cu::Blas & blas, cusparseHandle_t sparseHandle, const linalg::CsrMatrix<float> & cpuMatrix);

        // -> MSE of solution
        float solve(const int maxIters, const float target);
    };
}

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_GAUSSSEIDEL */
