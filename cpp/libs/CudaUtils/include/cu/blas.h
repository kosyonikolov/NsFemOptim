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

    void saxpy(Blas & blas, const int n, float * src, float * dst, float alpha);

    void scale(Blas & blas, const int n, float * dst, float alpha);
}

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_BLAS */
