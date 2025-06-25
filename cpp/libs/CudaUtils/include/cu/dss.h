#ifndef LIBS_CUDAUTILS_INCLUDE_CU_DSS
#define LIBS_CUDAUTILS_INCLUDE_CU_DSS

#include <string>

#include <cudss.h>

namespace cu
{
    class Dss
    {
        cudssHandle_t theHandle = 0;

    public:
        Dss();
        Dss(Dss const &) = delete;
        void operator=(Dss const &) = delete;

        ~Dss();

        cudssHandle_t handle()
        {
            return theHandle;
        }
    };

    std::string dssStatusName(const cudssStatus_t status);
}

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_DSS */
