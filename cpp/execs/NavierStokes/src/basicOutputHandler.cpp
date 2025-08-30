#include <NavierStokes/basicOutputHandler.h>

#include <filesystem>

#include <opencv2/opencv.hpp>

#include <mesh/interpolator.h>
#include <mesh/colorScale.h>
#include <mesh/drawMesh.h>

BasicOutputHandler::BasicOutputHandler(const OutputConfig & cfg,
                                       const mesh::ConcreteMesh & velocityMesh,
                                       const mesh::ConcreteMesh & pressureMesh)
    : cfg(cfg), velocityMesh(velocityMesh), pressureMesh(pressureMesh)
{

}

TimeStepOutput BasicOutputHandler::getCurrentOutput(const size_t iter, const float time)
{
    TimeStepOutput out;
    out.iter = iter;
    out.time = time;

    if (iter % cfg.frameStep == 0)
    {
        storedSteps.emplace_back();
        auto & last = storedSteps.back();
        out.velocity = &last.velocity;
        out.pressure = &last.pressure;
    }

    return out;
}

void BasicOutputHandler::finishOutput(const TimeStepOutput &)
{
    // Can't do anything - we have to wait for all frames to determine the pressure's scale
    // Perhaps if we want only the raw number we could save them here
}

void BasicOutputHandler::writeOutput(const std::string & outputDir)
{
    std::filesystem::create_directories(outputDir);

    // Find range of pressure
    float minP = std::numeric_limits<float>::infinity();
    float maxP = -std::numeric_limits<float>::infinity();
    const int nSteps = storedSteps.size();
    const int skipStart = cfg.pressureSkipSteps;
    // Don't consider the initial pressure levels - they will likely have a high pressure due to initial conditions
    for (int i = std::min(skipStart, std::max(nSteps - skipStart, 0)); i < nSteps; i++)
    {
        const auto & p = storedSteps[i].pressure;
        auto [minIt, maxIt] = std::minmax_element(p.begin(), p.end());
        minP = std::min(minP, *minIt);
        maxP = std::max(maxP, *maxIt);
    }
    maxP += 1e-3f;
    std::vector<cv::Scalar> colors{cv::Scalar(128, 0, 0), cv::Scalar(0, 0, 128), cv::Scalar(0, 200, 200)};
    mesh::SimpleColorScale pressureColorScale(minP, maxP, colors);

    mesh::TriangleLookup lookup(velocityMesh, 0.05);
    const float velocityStep = cfg.velocityStep;
    const float velocityScale = cfg.velocityScale / cfg.peakVelocity;
    const auto outputExt = cfg.ext;
    for (int i = 0; i < storedSteps.size(); i++)
    {
        const auto & s = storedSteps[i];
        const cv::Mat img = mesh::drawCfd(lookup, pressureColorScale, 800,
                                          velocityScale, velocityStep,
                                          velocityMesh, pressureMesh,
                                          s.velocity, s.pressure);
        const std::string outFname = std::format("{}/out_{}.{}", outputDir, i, outputExt);
        std::cout << outFname << "\n";
        cv::imwrite(outFname, img);
    }
}

BasicOutputHandler::~BasicOutputHandler()
{
}