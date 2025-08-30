#ifndef EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_BASICOUTPUTHANDLER
#define EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_BASICOUTPUTHANDLER

#include <mesh/concreteMesh.h>

#include <NavierStokes/abstractOutputHandler.h>
#include <NavierStokes/outputConfig.h>

class BasicOutputHandler : public AbstractOutputHandler
{
public:
    struct StoredStep
    {
        size_t iter;
        float time;
        std::vector<float> velocity; // [velocityX; velocityY]
        std::vector<float> pressure;
    };

private:
    OutputConfig cfg;

    const mesh::ConcreteMesh & velocityMesh;
    const mesh::ConcreteMesh & pressureMesh;

    std::vector<StoredStep> storedSteps;
    int storedStepId = 0;

public:
    BasicOutputHandler(const OutputConfig & cfg,
                       const mesh::ConcreteMesh & velocityMesh, 
                       const mesh::ConcreteMesh & pressureMesh);

    TimeStepOutput getCurrentOutput(const size_t iter, const float time);

    void finishOutput(const TimeStepOutput & output);

    void writeOutput(const std::string & outputDir);

    ~BasicOutputHandler();
};

#endif /* EXECS_NAVIERSTOKES_INCLUDE_NAVIERSTOKES_BASICOUTPUTHANDLER */
