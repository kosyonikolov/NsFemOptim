#ifndef LIBS_CUDAUTILS_INCLUDE_CU_DSSSOLVER
#define LIBS_CUDAUTILS_INCLUDE_CU_DSSSOLVER

#include <cudss.h>

#include <linalg/csrMatrix.h>

#include <cu/dss.h>
#include <cu/vec.h>

namespace cu
{
    class DssSolver
    {
        Dss & lib;

        // Size of matrix (N x N)
        int n;
        int numRhs;

        // size = nnz
        cu::vec<float> values;
        cu::vec<int> column;
        // size = rows + 1, last index is size(coeffs)
        cu::vec<int> rowStart;

        cudssConfig_t solverConfig;
        cudssData_t solverData;
        cudssMatrix_t xMat; // == sol
        cudssMatrix_t bMat; // == rhs
        cudssMatrix_t A;    // the CSR matrix

        bool hasAnalyzed = false;

    public:
        // Input/output vectors of size N x numRhs (column major)
        // Do not reallocate!
        cu::vec<float> rhs;
        cu::vec<float> sol;

        DssSolver(Dss & lib, const linalg::CsrMatrix<float> & m,
                  const int numRhs, const cudssMatrixType_t matrixType);

        // Analysis + factorization
        void analyze();

        void solve();
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_DSSSOLVER */
