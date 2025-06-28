#include <cu/stopwatch.h>

#include <format>
#include <stdexcept>

namespace cu
{
    Stopwatch::Stopwatch(cudaStream_t stream)
        : stream(stream)
    {
        auto rc = cudaEventCreate(&start);
        if (rc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("cudaEventCreate failed: {}", cudaGetErrorName(rc)));
        }

        rc = cudaEventCreate(&stop);
        if (rc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("cudaEventCreate failed: {}", cudaGetErrorName(rc)));
        }

        rc = cudaEventRecord(start, stream);
        if (rc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("cudaEventRecord failed: {}", cudaGetErrorName(rc)));
        }
    }

    void Stopwatch::reset()
    {
        auto rc = cudaEventRecord(start, stream);
        if (rc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("cudaEventRecord failed: {}", cudaGetErrorName(rc)));
        }
    }

    float Stopwatch::millis(const bool reset)
    {
        auto rc = cudaEventRecord(stop, stream);
        if (rc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("cudaEventRecord failed: {}", cudaGetErrorName(rc)));
        }

        rc = cudaEventSynchronize(stop);
        if (rc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("cudaEventSynchronize failed: {}", cudaGetErrorName(rc)));
        }

        float ms = -1;
        rc = cudaEventElapsedTime(&ms, start, stop);
        if (rc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("cudaEventElapsedTime failed: {}", cudaGetErrorName(rc)));
        }

        if (reset)
        {
            rc = cudaEventRecord(start, stream);
            if (rc != cudaError_t::cudaSuccess)
            {
                throw std::runtime_error(std::format("cudaEventRecord failed: {}", cudaGetErrorName(rc)));
            }
        }

        return ms;
    }
} // namespace cu