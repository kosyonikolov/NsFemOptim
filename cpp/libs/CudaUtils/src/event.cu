#include <cu/event.h>

#include <format>
#include <stdexcept>
#include <iostream>

namespace cu
{
    Event::Event()
    {
        auto rc = cudaEventCreate(&event);
        if (rc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("cudaEventCreate failed: {}", cudaGetErrorName(rc)));
        }
    }

    void Event::record()
    {
        auto rc = cudaEventRecord(event);
        if (rc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("cudaEventRecord failed: {}", cudaGetErrorName(rc)));
        }
    }

    void Event::sync()
    {
        auto rc = cudaEventSynchronize(event);
        if (rc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("cudaEventSynchronize failed: {}", cudaGetErrorName(rc)));
        }
    }

    float Event::elapsedTimeMs(Event & earlier)
    {
        float ms = -1;
        auto rc = cudaEventElapsedTime(&ms, earlier.event, event);
        if (rc != cudaError_t::cudaSuccess)
        {
            throw std::runtime_error(std::format("cudaEventElapsedTime failed: {}", cudaGetErrorName(rc)));
        }
        return ms;
    }

    Event::~Event()
    {
        auto rc = cudaEventDestroy(event);
        if (rc != cudaError_t::cudaSuccess)
        {
            std::cerr << "Failed to destroy event: " << rc << "\n";
        }
    }
}