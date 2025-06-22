#include <NavierStokes/buildContext.h>

#include <cassert>
#include <iostream>

#include <linalg/eigen.h>

#include <element/triangleIntegrator.h>

#include <fem/chorinCsr.h>
#include <fem/fastConvection.h>

#include <utils/stopwatch.h>

#include <NavierStokes/borders.h>

fem::ChorinContextF buildChorinContext(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                                       const DfgConditions & cond)
{
    assert(velocityMesh.groups.size() == pressureMesh.groups.size());
    assert(velocityMesh.numElements == pressureMesh.numElements);
    assert(velocityMesh.numBorderElements == pressureMesh.numBorderElements);

    const int idLeft = velocityMesh.findGroupId("Left");
    if (idLeft < 0)
    {
        throw std::invalid_argument("No left border!");
    }

    const int idRight = velocityMesh.findGroupId("Right");
    if (idRight < 0)
    {
        throw std::invalid_argument("No right border!");
    }

    const int idTop = velocityMesh.findGroupId("Top");
    if (idTop < 0)
    {
        throw std::invalid_argument("No top border!");
    }

    const int idBottom = velocityMesh.findGroupId("Bottom");
    if (idBottom < 0)
    {
        throw std::invalid_argument("No bottom border!");
    }

    const int idCircle = velocityMesh.findGroupId("Circle");
    if (idCircle < 0)
    {
        throw std::invalid_argument("No circle border!");
    }

    const int numVelocityNodes = velocityMesh.nodes.size(); // ! Counts only one channel, not X + Y - multiply by 2 to get all velocity nodes
    const int numPressureNodes = pressureMesh.nodes.size();

    fem::ChorinContextF result;
    result.numVelocityNodes = numVelocityNodes;
    result.numPressureNodes = numPressureNodes;

    // ======================================== Collect Dirichlet nodes ========================================
    std::cout << "Collecting Dirichlet nodes... ";
    std::cout.flush();
    auto dirichletZero = [](const mesh::ConcreteMesh &, const int, const int) -> float
    {
        return 0.0f;
    };

    auto calcDirichletVx = [&](const mesh::ConcreteMesh & mesh, const int nodeId, const int borderId) -> float
    {
        assert(borderId >= 0 && borderId < mesh.groups.size());
        const auto & group = mesh.groups[borderId];
        if (group == "Left")
        {
            const auto & node = mesh.nodes[nodeId];
            const float y = node.y;
            const float v = cond.calcLeftVelocity(y);
            return v;
        }
        return 0;
    };

    // Velocity
    const std::vector<int> velocityBorderIds = {idLeft, idTop, idBottom, idCircle};
    result.dirichletVx = extractDirichletNodes(velocityMesh, velocityBorderIds, calcDirichletVx);
    result.dirichletVy = extractDirichletNodes(velocityMesh, velocityBorderIds, dirichletZero);
    assert(result.dirichletVx.size() == result.dirichletVy.size());
    result.internalVelocityNodes = extractInternalNodes(numVelocityNodes, result.dirichletVx);

    // Pressure
    const std::vector<int> pressureBorderIds = {idRight};
    result.dirichletPressure = extractDirichletNodes(pressureMesh, pressureBorderIds, dirichletZero);
    result.internalPressureNodes = extractInternalNodes(numPressureNodes, result.dirichletPressure);
    // const int numInternalPressureNodes = result.internalPressureNodes.size();
    std::cout << "Done\n";
    // =========================================================================================================

    // =========================================== Assemble matrices ===========================================
    std::cout << "Assembling matrices... ";
    std::cout.flush();

    const int integrationDegree = 4;
    el::TriangleIntegrator velocityIntegrator(velocityMesh.baseElement, integrationDegree, pressureMesh.baseElement);

    u::Stopwatch bigSw;
    u::Stopwatch smallSw;
    const int nThreads = 8;
    auto chorinMats = fem::buildChorinCsrMatrices<float>(velocityMesh, pressureMesh, integrationDegree, nThreads);
    // clang-format off
    result.velocityMass        = std::move(chorinMats.velocityMass);
    result.velocityStiffness   = std::move(chorinMats.velocityStiffness);
    result.pressureStiffness   = std::move(chorinMats.pressureStiffness);
    result.velocityPressureDiv = std::move(chorinMats.velocityPressureDiv);
    result.pressureVelocityDiv = std::move(chorinMats.pressureVelocityDiv);
    // clang-format on
    std::cout << "Done\n";
    std::cout.flush();
    const auto tCsrs = smallSw.millis(true);

    result.pressureStiffnessInternal = result.pressureStiffness.slice(result.internalPressureNodes, result.internalPressureNodes);
    const auto tSlice = smallSw.millis(true);

    std::cout << "Setting up FastConvection... ";
    // TODO This is single threaded. Make a MT version
    // TODO Also pass the velocity stiffness matrix - no need to assemble fake convection (it has the same layout)
    fem::FastConvection fastConvection(velocityMesh, velocityIntegrator);
    std::cout << "Done\n";
    const auto tFastConvection = smallSw.millis(true);

    result.convection = linalg::csrFromEigen(fastConvection.convection);
    result.fastConvectionIntegration = linalg::csrFromEigen(fastConvection.integration);
    const auto tFcConvert = smallSw.millis(true);
    const auto tTotal = bigSw.millis();

    std::cout << "Context creation times:\n\tTotal = " << tTotal << "ms\n";
    std::cout << "\tAssemble matrices: " << tCsrs << " ms\n";
    std::cout << "\tSlice pressure: " << tSlice << " ms\n";
    std::cout << "\tSetup fast convection: " << tFastConvection << " ms\n";
    std::cout << "\tConvert fast convection: " << tFcConvert << " ms\n";

    return result;
}