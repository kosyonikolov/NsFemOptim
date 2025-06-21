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
} // namespace cu