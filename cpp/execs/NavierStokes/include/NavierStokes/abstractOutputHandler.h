#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_ABSTRACTOUTPUTHANDLER
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_ABSTRACTOUTPUTHANDLER

#include <vector>

struct TimeStepOutput
{
    size_t iter;
    float time;
    std::vector<float> * velocity;
    std::vector<float> * pressure;
};

class AbstractOutputHandler
{
public:
    // Request the output buffer for the current iteration and time
    // The output vectors may be nullptr, which indicates that the output is not required
    virtual TimeStepOutput getCurrentOutput(const size_t iter, const float time) = 0;

    // Signal that the output bundle is ready
    virtual void finishOutput(const TimeStepOutput & output) = 0;

    virtual ~AbstractOutputHandler(){};
};

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_ABSTRACTOUTPUTHANDLER */
