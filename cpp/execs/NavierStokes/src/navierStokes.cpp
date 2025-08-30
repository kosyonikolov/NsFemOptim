#include <cassert>
#include <iostream>
#include <string>

#include <mesh/drawMesh.h>
#include <mesh/io.h>

#include <element/factory.h>

#include <NavierStokes/basicOutputHandler.h>
#include <NavierStokes/buildContext.h>
#include <NavierStokes/chorinCuda.h>
#include <NavierStokes/chorinEigen.h>
#include <NavierStokes/nsConfig.h>
#include <NavierStokes/solution.h>

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
    cfg.output.peakVelocity = cfg.peakVelocity;

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

    const float tau = cfg.tau;
    const float maxT = cfg.maxT;

    BasicOutputHandler outputHandler(cfg.output, outputDir, velocityMesh, pressureMesh);

    std::cout << "Algorithm = " << cfg.algo << "\n";
    if (cfg.algo == "chorinEigen")
    {
        std::cout << "Using CPU Eigen-based Chorin method\n";
        solveNsChorinEigen(velocityMesh, pressureMesh, cond, tau, maxT, outputHandler);
    }
    else if (cfg.algo == "chorinCuda")
    {
        std::cout << "Using CUDA Chorin method\n";
        solveNsChorinCuda(velocityMesh, pressureMesh, cond, tau, maxT, cfg.chorinCuda, outputHandler);
    }

    outputHandler.writeOutput();

    return 0;
}
