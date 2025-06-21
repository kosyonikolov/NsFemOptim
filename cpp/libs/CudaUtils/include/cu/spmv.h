#ifndef LIBS_CUDAUTILS_INCLUDE_CU_SPMV
#define LIBS_CUDAUTILS_INCLUDE_CU_SPMV

#include <cusparse.h>

#include <linalg/csrMatrix.h>

#include <cu/vec.h>
#include <cu/csrF.h>

namespace cu
{
    struct spmv
    {
        constexpr static auto op = cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE;

        cusparseHandle_t handle;

        // ========== Matrix data ==========
        cu::csrF & mat;
        cusparseSpMatDescr_t matDesc;

        // ========== SPMV data ==========
        // b = Mx
        // These are the "default" operands for the multiplication
        // Other vectors may also be used
        cu::vec<float> x;
        cu::vec<float> b;

        cu::vec<char> workspace;

        float alpha = 1.0f;
        float beta = 0.0f;

        spmv(cusparseHandle_t handle, cu::csrF & m);

        ~spmv();

        // Compute b = Mx using the internal vectors
        void compute();

        // Compute b = Mx using the supplied vectors
        void compute(cusparseDnVecDescr_t otherX, cusparseDnVecDescr_t otherB);

        void compute(cu::vec<float> & otherX, cu::vec<float> & otherB);
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_SPMV */
