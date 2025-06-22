#include <format>
#include <stdexcept>
#include <cassert>
#include <iostream>

#include <cu/blas.h>

namespace cu
{
    Blas::Blas()
    {
        auto rc = cublasCreate(&handle);
        if (rc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create cublas handle: {}", cublasGetStatusName(rc)));
        }
    }

    Blas::~Blas()
    {
        if (handle)
        {
            auto rc = cublasDestroy_v2(handle);
            if (rc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
            {
                std::cerr << "Failed to destroy cublas handle: " << cublasGetStatusName(rc) << "\n";
            }
        }
    }

    void Blas::setStream(cudaStream_t stream)
    {
        assert(handle);
        auto rc = cublasSetStream(handle, stream);
        if (rc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to set cublas stream: {}", cublasGetStatusName(rc)));
        }
    }

    void Blas::setPointerMode(cublasPointerMode_t pointerMode)
    {
        assert(handle);
        auto rc = cublasSetPointerMode(handle, pointerMode);
        if (rc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to set cublas pointer mode: {}", cublasGetStatusName(rc)));
        }
    }

    void saxpy(Blas & blas, const int n, float * src, float * dst, float alpha)
    {
        auto rc = cublasSaxpy(blas.handle, n, &alpha, src, 1, dst, 1);
        if (rc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cublasSaxpy failed: {}", cublasGetStatusName(rc)));
        }
    }

    void scale(Blas & blas, const int n, float * dst, float alpha)
    {
        auto rc = cublasSscal(blas.handle, n, &alpha, dst, 1);
        if (rc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cublasSscal failed: {}", cublasGetStatusName(rc)));
        }
    }
} // namespace cu