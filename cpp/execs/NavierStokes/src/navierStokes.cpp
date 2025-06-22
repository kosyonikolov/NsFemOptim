#include <cassert>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>

#include <mesh/colorScale.h>
#include <mesh/drawMesh.h>
#include <mesh/io.h>
#include <mesh/triangleLookup.h>

#include <element/factory.h>

#include <NavierStokes/chorinEigen.h>
#include <NavierStokes/chorinCuda.h>
#include <NavierStokes/nsConfig.h>
#include <NavierStokes/solution.h>
#include <NavierStokes/buildContext.h>

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./NavierStokes <config> <msh file> <output dir>";
    if (argc != 4)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string cfgFname = argv[1];
    const std::string meshFileName = argv[2];
    const std::string outputDir = argv[3];

    auto cfg = parseNsConfig(cfgFname);

    std::cout << "Parsing mesh... ";
    std::cout.flush();
    auto triMesh = mesh::parseTriangleGmsh(meshFileName);
    std::cout << "Done\n";

    const auto velocityElement = el::createElement(el::Type::P2);
    const auto pressureElement = el::createElement(el::Type::P1);

    std::cout << "Creating pressure and velocity meshes... ";
    std::cout.flush();
    auto velocityMesh = mesh::createMesh(triMesh, *velocityElement);
    auto pressureMesh = mesh::createMesh(triMesh, *pressureElement);
    std::cout << "Done\n";

    if (false)
    {
        const float scale = 3500;
        cv::imwrite("velocity_mesh.png", mesh::drawMesh(velocityMesh, scale));
        cv::imwrite("pressure_mesh.png", mesh::drawMesh(pressureMesh, scale));
    }

    DfgConditions cond;
    cond.viscosity = cfg.viscosity;
    cond.peakVelocity = cfg.peakVelocity;

    // const auto context = buildChorinContext(velocityMesh, pressureMesh, cond);
    // {
    //     const bool ok = context.velocityStiffness.compareLayout(context.convection);
    //     if (!ok)
    //     {
    //         std::cerr << "Velocity stiffness has different layout compared to convection!\n";
    //         return 1;
    //     }
    // }

    const float tau = cfg.tau;
    const float maxT = cfg.maxT;
    Solution sol;
    std::cout << "Algorithm = " << cfg.algo << "\n";
    if (cfg.algo == "chorinEigen")
    {
        std::cout << "Using CPU Eigen-based Chorin method\n";
        sol = solveNsChorinEigen(velocityMesh, pressureMesh, cond, tau, maxT);
    }
    else if (cfg.algo == "chorinCuda")
    {
        std::cout << "Using CUDA Chorin method\n";
        sol = solveNsChorinCuda(velocityMesh, pressureMesh, cond, tau, maxT);
    }

    // Find range of pressure
    float minP = std::numeric_limits<float>::infinity();
    float maxP = -std::numeric_limits<float>::infinity();
    const int nSteps = sol.steps.size();
    const int skipStart = 5;
    // Don't consider the initial pressure levels - they will likely have a high pressure due to initial conditions
    for (int i = std::min(skipStart, std::max(nSteps - skipStart, 0)); i < nSteps; i++)
    {
        const auto & p = sol.steps[i].pressure;
        auto [minIt, maxIt] = std::minmax_element(p.begin(), p.end());
        minP = std::min(minP, *minIt);
        maxP = std::max(maxP, *maxIt);
    }
    maxP += 1e-3f;
    std::vector<cv::Scalar> colors{cv::Scalar(128, 0, 0), cv::Scalar(0, 0, 128), cv::Scalar(0, 200, 200)};
    mesh::SimpleColorScale pressureColorScale(minP, maxP, colors);

    std::filesystem::create_directories(outputDir);

    mesh::TriangleLookup lookup(velocityMesh, 0.05);
    const float velocityStep = cfg.output.velocityStep;
    const float velocityScale = cfg.output.velocityScale / cfg.peakVelocity;
    int j = 0;
    for (int i = 0; i < sol.steps.size(); i += cfg.output.frameStep, j++)
    {
        const auto & s = sol.steps[i];
        const cv::Mat img = mesh::drawCfd(lookup, pressureColorScale, 800,
                                          velocityScale, velocityStep,
                                          velocityMesh, pressureMesh,
                                          s.velocity, s.pressure);
        const std::string outFname = std::format("{}/out_{}.png", outputDir, j);
        std::cout << outFname << "\n";
        cv::imwrite(outFname, img);
    }

    return 0;
}
