#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_DFGCONDTIONS
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_DFGCONDTIONS

struct DfgConditions
{
    float viscosity;
    float peakVelocity;

    float calcLeftVelocity(const float y) const
    {
        return peakVelocity * y * (0.41 - y) / (0.41 * 0.41);
    }
};

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_DFGCONDTIONS */
