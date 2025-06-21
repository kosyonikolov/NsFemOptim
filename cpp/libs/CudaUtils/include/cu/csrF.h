#ifndef LIBS_CUDAUTILS_INCLUDE_CU_CSRF
#define LIBS_CUDAUTILS_INCLUDE_CU_CSRF

#include <cusparse.h>

#include <linalg/csrMatrix.h>

#include <cu/vec.h>

namespace cu
{
    struct csrF
    {
        int rows, cols;

        // size = nnz
        cu::vec<float> values; // Can be modified after creating the matrix
        cu::vec<int> column;
        // size = rows + 1, last index is size(coeffs)
        cu::vec<int> rowStart;
        
        cusparseSpMatDescr_t matDesc = 0;

        csrF(const linalg::CsrMatrix<float> & cpuMat);

        ~csrF();

        // Get descriptor, create if necessary
        cusparseSpMatDescr_t getCuSparseDescriptor();
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_CSRF */
