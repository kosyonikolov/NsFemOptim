#ifndef LIBS_CUDAUTILS_INCLUDE_CU_CUSPARSE
#define LIBS_CUDAUTILS_INCLUDE_CU_CUSPARSE

#include <cusparse.h>

namespace cu
{
    class Sparse
    {
        cusparseHandle_t theHandle;

        Sparse();

    public:
        Sparse(Sparse const &) = delete;
        void operator=(Sparse const &) = delete;

        ~Sparse();

        static Sparse & instance()
        {
            static Sparse theInstance;
            return theInstance;
        }

        cusparseHandle_t handle()
        {
            return theHandle;
        }
    };

    cusparseHandle_t getCuSparseHandle();
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_CUSPARSE */
