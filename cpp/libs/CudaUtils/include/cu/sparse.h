#ifndef LIBS_CUDAUTILS_INCLUDE_CU_CUSPARSE
#define LIBS_CUDAUTILS_INCLUDE_CU_CUSPARSE

#include <cusparse.h>

namespace cu
{
    class Sparse
    {
        cusparseHandle_t theHandle;

    public:
        Sparse();
        Sparse(Sparse const &) = delete;
        void operator=(Sparse const &) = delete;

        ~Sparse();

        cusparseHandle_t handle()
        {
            return theHandle;
        }
    };
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_CUSPARSE */
