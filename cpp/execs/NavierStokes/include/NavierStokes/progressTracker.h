#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_PROGRESSTRACKER
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_PROGRESSTRACKER

#include <utils/stopwatch.h>

class ProgressTracker
{
    u::Stopwatch globalSw;

    float printIntervalMs;
    float lastPrintTime;
    int numTimeSteps;

public:
    ProgressTracker(const int numTimeSteps, const float printIntervalMs = 500);

    void update(const int iteration);
};

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_PROGRESSTRACKER */
