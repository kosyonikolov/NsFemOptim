#ifndef LIBS_CUDAUTILS_INCLUDE_CU_CSRF
#define LIBS_CUDAUTILS_INCLUDE_CU_CSRF

#include <cusparse.h>

#include <linalg/csrMatrix.h>

#include <cu/vec.h>

namespace cu
{
    struct csrF
    {
        constexpr static auto op = cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE;

        cusparseHandle_t handle;

        // ========== Matrix data ==========
        int rows, cols;

        // size = nnz
        cu::vec<float> values; // Can be modified after creating the matrix
        cu::vec<int> column;
        // size = rows + 1, last index is size(coeffs)
        cu::vec<int> rowStart;
        
        cusparseSpMatDescr_t matDesc;

        // ========== SPMV data ==========
        // b = Mx
        // These are the "default" operands for the multiplication
        // Other vectors may also be used
        cu::vec<float> x;
        cu::vec<float> b;
        cusparseDnVecDescr_t xDesc, bDesc;

        cu::vec<char> workspace;

        float alpha = 1.0f;
        float beta = 0.0f;

        csrF(const linalg::CsrMatrix<float> & cpuMat);

        ~csrF();

        // Compute b = Mx using the internal vectors
        void spmv();

        // Compute b = Mx using the supplied vectors
        void spmv(cusparseDnVecDescr_t otherX, cusparseDnVecDescr_t otherB);
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_CSRF */
