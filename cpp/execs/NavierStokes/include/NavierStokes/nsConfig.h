#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_NSCONFIG
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_NSCONFIG

#include <string>

#include <NavierStokes/chorinCudaConfig.h>
#include <NavierStokes/outputConfig.h>

struct NsConfig
{
    std::string algo = "chorinEigen";

    float viscosity = 0.001;
    float peakVelocity = 1;
    float maxT = 1;
    float tau = 1e-4;

    float rampTime = 0;

    ChorinCudaConfig chorinCuda;

    OutputConfig output;   
};

NsConfig parseNsConfig(const std::string & fileName);

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_NSCONFIG */
