#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_SOLUTION
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_SOLUTION

#include <vector>

struct TimeStepSolution
{
    float time;
    std::vector<float> velocity; // [velocityX; velocityY]
    std::vector<float> pressure;
};

struct Solution
{
    std::vector<TimeStepSolution> steps;
};


#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_SOLUTION */
