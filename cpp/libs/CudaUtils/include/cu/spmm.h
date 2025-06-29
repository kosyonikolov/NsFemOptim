#ifndef LIBS_CUDAUTILS_INCLUDE_CU_SPMM
#define LIBS_CUDAUTILS_INCLUDE_CU_SPMM

#include <cusparse.h>

#include <linalg/csrMatrix.h>

#include <cu/csrF.h>
#include <cu/vec.h>

namespace cu
{
    // For Mx=b, where x and b are column-major matrices
    struct spmm
    {
        constexpr static auto op = cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE;

        cusparseHandle_t handle;

        // ========== Matrix data ==========
        cu::csrF & mat;
        cusparseSpMatDescr_t matDesc;

        // ========== SPMV data ==========
        int numCh;

        // b = Mx
        // These are the "default" operands for the multiplication
        // Other vectors may also be used
        cu::vec<float> x;
        cu::vec<float> b;

        cu::vec<char> workspace;

        float alpha = 1.0f;
        float beta = 0.0f;

        spmm(cusparseHandle_t handle, cu::csrF & m, const int numCh);

        ~spmm();

        // Compute b = Mx using the internal vectors
        void compute();

        // Compute b = Mx using the supplied vectors
        void compute(cusparseDnMatDescr_t otherX, cusparseDnMatDescr_t otherB);

        void compute(cu::vec<float> & otherX, cu::vec<float> & otherB);
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_SPMM */
