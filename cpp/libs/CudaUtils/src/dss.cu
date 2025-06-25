#include <cu/dss.h>

#include <stdexcept>
#include <format>
#include <iostream>

namespace cu
{
    Dss::Dss()
    {
        auto rc = cudssCreate(&theHandle);
        if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create cuDSS handle: {}", dssStatusName(rc)));
        }
    }

    Dss::~Dss()
    {
        if (theHandle)
        {
            auto rc = cudssDestroy(theHandle);
            if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
            {
                std::cerr << "Failed to destroy cuDSS handle: " << dssStatusName(rc) << "\n";
            }
            theHandle = 0;
        }
    }

    std::string dssStatusName(const cudssStatus_t status)
    {
        #define RETIF(x) if (status == cudssStatus_t::x) return #x
        RETIF(CUDSS_STATUS_SUCCESS);
        RETIF(CUDSS_STATUS_NOT_INITIALIZED);
        RETIF(CUDSS_STATUS_ALLOC_FAILED);
        RETIF(CUDSS_STATUS_INVALID_VALUE);
        RETIF(CUDSS_STATUS_NOT_SUPPORTED);
        RETIF(CUDSS_STATUS_EXECUTION_FAILED);
        RETIF(CUDSS_STATUS_INTERNAL_ERROR);
        #undef RETIF

        return std::format("Unknown [{}]", static_cast<int>(status));
    }
}