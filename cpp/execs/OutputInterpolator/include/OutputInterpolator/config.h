#ifndef EXECS_OUTPUTINTERPOLATOR_INCLUDE_OUTPUTINTERPOLATOR_CONFIG
#define EXECS_OUTPUTINTERPOLATOR_INCLUDE_OUTPUTINTERPOLATOR_CONFIG

#include <string>

struct Config
{
    std::string ext = "ppm";
    float imgScale = 800;

    float velocityStep = 0.025;
    float velocityScale = 0.0025;

    int pressureSkipSteps = 5;

    int numThreads = 8;

    bool dumpPressureStats = false;
    bool manualPressure = false;
    float minPressure = 0;
    float maxPressure = 1;

    bool sequentialOutput = false;
};

Config parseConfig(const std::string & fileName);

#endif /* EXECS_OUTPUTINTERPOLATOR_INCLUDE_OUTPUTINTERPOLATOR_CONFIG */
