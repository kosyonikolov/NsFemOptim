#ifndef LIBS_CUDAUTILS_INCLUDE_CU_BLAS
#define LIBS_CUDAUTILS_INCLUDE_CU_BLAS

#include <cublas_v2.h>

namespace cu
{
    struct Blas
    {
        cublasHandle_t handle = 0;

        Blas();
        
        Blas(const Blas & other) = delete;
        Blas & operator=(const Blas & other) = delete;

        ~Blas();

        void setStream(cudaStream_t stream);

        void setPointerMode(cublasPointerMode_t pointerMode);
    };
}

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_BLAS */
