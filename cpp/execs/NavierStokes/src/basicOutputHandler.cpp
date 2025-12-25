#include <NavierStokes/basicOutputHandler.h>

#include <filesystem>

#include <opencv2/opencv.hpp>

#include <linalg/io.h>

#include <mesh/interpolator.h>
#include <mesh/colorScale.h>
#include <mesh/drawMesh.h>

BasicOutputHandler::BasicOutputHandler(const OutputConfig & cfg, const std::string & outputDir,
                                       const mesh::ConcreteMesh & velocityMesh,
                                       const mesh::ConcreteMesh & pressureMesh)
    : cfg(cfg), outputDir(outputDir), velocityMesh(velocityMesh), pressureMesh(pressureMesh)
{
    std::filesystem::create_directories(outputDir);

    if (cfg.writeBinary && !cfg.writeImages)
    {
        // We need only one stored step - we're going to write it to disk as soon as it's ready
        storedSteps.emplace_back();
    }
}

TimeStepOutput BasicOutputHandler::getCurrentOutput(const size_t iter, const float time)
{
    TimeStepOutput out;
    out.iter = iter;
    out.time = time;
    out.pressure = 0;
    out.velocity = 0;

    if (!cfg.writeBinary && !cfg.writeImages)
    {
        // No output at all
        return out;
    }

    if (time < cfg.skipTime)
    {
        return out;
    }

    if (iter % cfg.frameStep == 0)
    {
        if (cfg.writeImages)
        {
            // If we want output images, we need to store all frames
            storedSteps.emplace_back();
        }
        auto & last = storedSteps.back();
        out.velocity = &last.velocity;
        out.pressure = &last.pressure;
    }

    return out;
}

void BasicOutputHandler::finishOutput(const TimeStepOutput & output)
{
    const bool haveVelocity = output.velocity != 0;
    const bool havePressure = output.pressure != 0;
    if (cfg.writeBinary && haveVelocity && havePressure)
    {
        const std::string outFnameVelocity = std::format("{}/velocity_{}.bin", outputDir, storedStepId);
        const std::string outFnamePressure = std::format("{}/pressure_{}.bin", outputDir, storedStepId);
        storedStepId++;
        // std::cout << outFnameVelocity << ", " << outFnamePressure << "\n";
        linalg::write(outFnameVelocity, *output.velocity);
        linalg::write(outFnamePressure, *output.pressure);
    }
}

void BasicOutputHandler::writeOutput()
{
    if (!cfg.writeImages)
    {
        return;
    }

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