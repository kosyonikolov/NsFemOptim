#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINCUDACONFIG
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINCUDACONFIG

#include <string>

struct SolverConfig
{
    std::string method = "gs";
    int maxIterations = 200;
    float targetMse = 1e-6f;
    int mseCheckInterval = 1;
};

struct ChorinCudaConfig
{
    SolverConfig velocitySolver;
    SolverConfig pressureSolver;
};

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_CHORINCUDACONFIG */
