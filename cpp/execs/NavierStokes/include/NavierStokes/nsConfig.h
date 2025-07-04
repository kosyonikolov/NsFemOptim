#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_NSCONFIG
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_NSCONFIG

#include <string>

#include <NavierStokes/chorinCudaConfig.h>

struct NsConfig
{
    std::string algo = "chorinEigen";

    float viscosity = 0.001;
    float peakVelocity = 1;
    float maxT = 1;
    float tau = 1e-4;

    ChorinCudaConfig chorinCuda;

    struct
    {
        std::string ext = "ppm";
        int frameStep = 1;
        float velocityStep = 0.025;
        float velocityScale = 0.05;
        float imgScale = 800;
    } output;   
};

NsConfig parseNsConfig(const std::string & fileName);

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_NSCONFIG */
