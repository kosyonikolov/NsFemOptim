#include <iostream>
#include <string>
#include <format>

#include <mesh/io.h>
#include <mesh/concreteMesh.h>

#include <element/factory.h>

#include <fem/chorinMatrices.h>
#include <fem/chorinCsr.h>

#include <linalg/csrMatrix.h>

#include <utils/stopwatch.h>

using SpMat = linalg::CsrMatrix<float>;

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./AssemblyTest <mesh file>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string meshFile = argv[1];

    const auto triMesh = mesh::parseTriangleGmsh(meshFile);
    std::cout << std::format("Nodes = {}, elements = {}\n", triMesh.nodes.size(), triMesh.elements.size());

    const auto velocityElement = el::createElement(el::Type::P2);
    const auto pressureElement = el::createElement(el::Type::P0);
    const int integrationDegree = 4;

    const auto velocityMesh = mesh::createMesh(triMesh, *velocityElement);
    const auto pressureMesh = mesh::createMesh(triMesh, *pressureElement);

    std::cout << "Test bucket building\n";
    SpMat velocityMass, velocityStiffness;
    SpMat pressureStiffness;
    SpMat vpDiv, pvDiv;

    const int nRuns = 5;
    for (int i = 0; i < nRuns; i++)
    {
        u::Stopwatch bigSw;
        u::Stopwatch sw;
        auto proto = fem::buildChorinMatrices<float>(velocityMesh, pressureMesh, integrationDegree);
        const auto tProto = sw.millis(true);

        velocityMass = proto.velocityMass.buildCsr();
        velocityStiffness = proto.velocityStiffness.buildCsr();
        pressureStiffness = proto.pressureStiffness.buildCsr();
        vpDiv = proto.velocityPressureDiv.buildCsr();
        pvDiv = proto.pressureVelocityDiv.buildCsr();
        const auto tBuild = sw.millis();
        const auto tTotal = bigSw.millis();

        std::cout << std::format("{}: total = {}, proto = {}, build = {}\n", i, tTotal, tProto, tBuild);
    }

    std::cout << "Test CSR building\n";
    const int nThreads = 8;
    SpMat velocityMassTest, velocityStiffnessTest;
    SpMat pressureStiffnessTest;
    SpMat vpDivTest, pvDivTest;
    for (int i = 0; i < nRuns; i++)
    {
        u::Stopwatch bigSw;
        u::Stopwatch sw;

        auto res = fem::buildChorinCsrMatrices<float>(velocityMesh, pressureMesh, integrationDegree, nThreads);
        const auto tTotal = bigSw.millis();

        velocityMassTest = std::move(res.velocityMass);
        velocityStiffnessTest = std::move(res.velocityStiffness);
        pressureStiffnessTest = std::move(res.pressureStiffness);
        vpDivTest = std::move(res.velocityPressureDiv);
        pvDivTest = std::move(res.pressureVelocityDiv);

        std::cout << std::format("{}: total = {}\n", i, tTotal);
    }

    // Make sure it's correct
    const auto cmp = [](const linalg::CsrMatrix<float> & a, const linalg::CsrMatrix<float> & b)
    {
        if (!a.compareLayout(b))
        {
            std::cerr << "LAYOUT\n";
            return false;
        }
        if (!a.compareValues(b, 1e-6f))
        {
            std::cerr << "VALUES\n";
            return false;
        }
        return true;
    };

    if (!cmp(velocityMassTest, velocityMass))
    {
        std::cerr << "Bad velocityMass\n";
    }
    if (!cmp(velocityStiffnessTest, velocityStiffness))
    {
        std::cerr << "Bad velocityStiffness\n";
    }
    if (!cmp(pressureStiffnessTest, pressureStiffness))
    {
        std::cerr << "Bad pressureStiffness\n";
    }
    if (!cmp(vpDivTest, vpDiv))
    {
        std::cerr << "Bad vpDiv\n";
    }
    if (!cmp(pvDivTest, pvDiv))
    {
        std::cerr << "Bad pvDiv\n";
    }

    return 0;
}