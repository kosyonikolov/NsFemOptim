#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_OUTPUTCONFIG
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_OUTPUTCONFIG

#include <string>

struct OutputConfig
{
    std::string ext = "ppm";
    int frameStep = 1;
    float velocityStep = 0.025;
    float velocityScale = 0.05;
    float imgScale = 800;

    int pressureSkipSteps = 5;

    // Fields that aren't parsed
    float peakVelocity; // Copied from the main config
};

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_OUTPUTCONFIG */
