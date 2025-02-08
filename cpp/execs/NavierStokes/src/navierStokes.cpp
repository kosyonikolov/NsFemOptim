#include <cassert>
#include <format>
#include <iostream>
#include <string>

#include <Eigen/Sparse>

#include <element/affineTransform.h>
#include <element/triangleIntegrator.h>
#include <element/factory.h>

#include <mesh/colorScale.h>
#include <mesh/concreteMesh.h>
#include <mesh/drawMesh.h>
#include <mesh/gmsh.h>
#include <mesh/interpolator.h>
#include <mesh/io.h>
#include <mesh/triangleLookup.h>

using SolType = double;
using SpMat = Eigen::SparseMatrix<SolType>;
using Triplet = Eigen::Triplet<SolType>;
using Vector = Eigen::Vector<SolType, Eigen::Dynamic>;

struct DirichletNode
{
    int id;
    float value;
    bool operator<(const DirichletNode & other) const
    {
        return id < other.id;
    }
};

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

float calcDirichletVx(const mesh::ConcreteMesh & mesh, const int nodeId, const int borderId)
{
    assert(borderId >= 0 && borderId < mesh.groups.size());
    const auto & group = mesh.groups[borderId];
    if (group == "Left")
    {
        const auto & node = mesh.nodes[nodeId];
        const float y = node.y;
        const float v = 20 * y * (0.41 - y) / (0.41 * 0.41);
        return v;
    }
    return 0;
}

float calcDirichletVy(const mesh::ConcreteMesh &, const int, const int)
{
    return 0;
}

float calcDirichletPressure(const mesh::ConcreteMesh &, const int, const int)
{
    return 0;
}

template <typename BorderValueFn>
std::vector<DirichletNode> extractDirichletNodes(const mesh::ConcreteMesh & mesh,
                                                 const std::vector<int> borderIds,
                                                 BorderValueFn borderValueFn)
{
    std::vector<DirichletNode> result;

    const int elSize = mesh.getBorderElementSize();
    const int numBorderElems = mesh.numBorderElements;
    std::vector<int> ptIds(elSize);
    std::vector<bool> seen(mesh.nodes.size(), false);
    for (int i = 0; i < numBorderElems; i++)
    {
        int triangle, side, group;
        mesh.getBorderElement(i, triangle, side, group, ptIds.data(), 0);
        auto it = std::find(borderIds.begin(), borderIds.end(), group);
        if (it == borderIds.end())
        {
            continue;
        }
        for (int k = 0; k < elSize; k++)
        {
            const int nodeIdx = ptIds[k];
            if (seen[nodeIdx])
            {
                continue;
            }
            const float val = borderValueFn(mesh, nodeIdx, group);
            seen[nodeIdx] = true;
            result.push_back(DirichletNode{ptIds[k], val});
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

std::vector<int> extractInternalNodes(const int numNodes, const std::vector<DirichletNode> & sortedDirichletNodes)
{
    assert(std::is_sorted(sortedDirichletNodes.begin(), sortedDirichletNodes.end()));
    std::vector<int> result;

    int i = 0;
    int j = 0;
    while (i < numNodes && j < sortedDirichletNodes.size())
    {
        const int d = sortedDirichletNodes[j].id;
        if (i < d)
        {
            result.push_back(i);
            i++;
        }
        else if (i == d)
        {
            i++;
            j++;
        }
        else if (d < i)
        {
            j++;
        }
    }

    while (i < numNodes)
    {
        result.push_back(i);
        i++;
    }

    return result;
}

std::vector<Triplet> projectTriplets(const int numNodes, const std::vector<Triplet> & orig, const std::vector<int> & newIds)
{
    std::vector<int> remap(numNodes, -1); // Old to new idx
    for (int i = 0; i < newIds.size(); i++)
    {
        const int j = newIds[i];
        remap[j] = i;
    }

    std::vector<Triplet> result;
    for (const Triplet & t : orig)
    {
        const int i = remap[t.row()];
        const int j = remap[t.col()];
        const auto v = t.value();
        if (i < 0 || j < 0)
        {
            continue;
        }
        result.push_back(Triplet(i, j, v));
    }

    return result;
};

Solution solveNsChorinEigen(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                            const float timeStep0, const float maxT)
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

    // ======================================== Collect Dirichlet nodes ========================================
    // Velocity
    const std::vector<int> velocityBorderIds = {idLeft, idTop, idBottom, idCircle};
    const auto dirichletVx = extractDirichletNodes(velocityMesh, velocityBorderIds, calcDirichletVx);
    const auto dirichletVy = extractDirichletNodes(velocityMesh, velocityBorderIds, calcDirichletVy);
    assert(dirichletVx.size() == dirichletVy.size());
    const auto internalVelocityNodes = extractInternalNodes(numVelocityNodes, dirichletVx);

    // Pressure
    const std::vector<int> pressureBorderIds = {idRight};
    const auto dirichletPressure = extractDirichletNodes(pressureMesh, pressureBorderIds, calcDirichletPressure);
    const auto internalPressureNodes = extractInternalNodes(numPressureNodes, dirichletPressure);
    const int numInternalPressureNodes = internalPressureNodes.size();
    // =========================================================================================================

    // =========================================== Assemble matrices ===========================================
    std::vector<Triplet> velocityMassT, velocityStiffnessT;
    std::vector<Triplet> pressureStiffnessT, pressureStiffnessInternalT;

    // X on top, Y on bottom
    std::vector<Triplet> velocityPressureDivT;
    std::vector<Triplet> pressureVelocityDivT;

    std::vector<Triplet> convectionT;

    SpMat velocityMass(numVelocityNodes, numVelocityNodes);
    SpMat velocityStiffness(numVelocityNodes, numVelocityNodes);
    SpMat convection(numVelocityNodes, numVelocityNodes);
    SpMat pressureInternalStiffness(numInternalPressureNodes, numInternalPressureNodes);
    SpMat velocityPressureDiv(numPressureNodes, 2 * numVelocityNodes);
    SpMat pressureVelocityDiv(2 * numVelocityNodes, numPressureNodes);

    const int integrationDegree = 4;
    el::TriangleIntegrator velocityIntegrator(velocityMesh.baseElement, integrationDegree, pressureMesh.baseElement);
    el::TriangleIntegrator pressureIntegrator(pressureMesh.baseElement, integrationDegree);

    const int elSizeV = velocityMesh.getElementSize();
    const int elSizeP = pressureMesh.getElementSize();

    std::vector<int> idsV(elSizeV);
    std::vector<el::Point> ptsV(elSizeV);
    cv::Mat localMassMatrixV, localStiffnessMatrixV;

    std::vector<int> idsP(elSizeP);
    std::vector<el::Point> ptsP(elSizeP);
    cv::Mat localMassMatrixP, localStiffnessMatrixP;

    cv::Mat localVpdX, localVpdY, localPvdX, localPvdY;

    const int nElem = velocityMesh.numElements;
    for (int i = 0; i < nElem; i++)
    {
        const auto t = velocityMesh.elementTransforms[i];

        // ========================= Velocity-only matrices =========================
        velocityMesh.getElement(i, idsV.data(), ptsV.data());

        velocityIntegrator.integrateLocalMassMatrix(t, localMassMatrixV);
        assert(elSizeV == localMassMatrixV.cols);
        assert(elSizeV == localMassMatrixV.rows);

        velocityIntegrator.integrateLocalStiffnessMatrix(t, localStiffnessMatrixV);
        assert(elSizeV == localStiffnessMatrixV.cols);
        assert(elSizeV == localStiffnessMatrixV.rows);

        // Accumulate
        for (int r = 0; r < elSizeV; r++)
        {
            const int i = idsV[r];
            for (int c = 0; c < elSizeV; c++)
            {
                const int j = idsV[c];
                const Triplet tripletM(i, j, localMassMatrixV.at<float>(r, c));
                const Triplet tripletS(i, j, localStiffnessMatrixV.at<float>(r, c));
                velocityMassT.push_back(tripletM);
                velocityStiffnessT.push_back(tripletS);
            }
        }

        // ========================= Pressure-only matrices =========================
        pressureMesh.getElement(i, idsP.data(), ptsP.data());

        pressureIntegrator.integrateLocalStiffnessMatrix(t, localStiffnessMatrixP);
        assert(elSizeP == localStiffnessMatrixP.cols);
        assert(elSizeP == localStiffnessMatrixP.rows);

        // Accumulate
        for (int r = 0; r < elSizeP; r++)
        {
            const int i = idsP[r];
            for (int c = 0; c < elSizeP; c++)
            {
                const int j = idsP[c];
                const Triplet tripletS(i, j, localStiffnessMatrixP.at<float>(r, c));
                pressureStiffnessT.push_back(tripletS);
            }
        }

        // ========================= Divergence matrices =========================
        velocityIntegrator.integrateLocalDivergenceMatrix(t, false, localVpdX, localVpdY);
        assert(localVpdX.cols == localVpdY.cols && localVpdX.rows == localVpdY.rows);
        assert(localVpdX.rows == elSizeP);
        assert(localVpdX.cols == elSizeV);

        velocityIntegrator.integrateLocalDivergenceMatrix(t, true, localPvdX, localPvdY);
        assert(localPvdX.cols == localPvdY.cols && localPvdX.rows == localPvdY.rows);
        assert(localPvdX.rows == elSizeV);
        assert(localPvdX.cols == elSizeP);

        for (int iV = 0; iV < elSizeV; iV++)
        {
            const int gV = idsV[iV];
            for (int iP = 0; iP < elSizeP; iP++)
            {
                const int gP = idsP[iP];
                // clang-format off
                velocityPressureDivT.emplace_back(gP, gV,                    localVpdX.at<float>(iP, iV));
                velocityPressureDivT.emplace_back(gP, gV + numVelocityNodes, localVpdY.at<float>(iP, iV));
                pressureVelocityDivT.emplace_back(gV, gP,                    localPvdX.at<float>(iV, iP));
                pressureVelocityDivT.emplace_back(gV + numVelocityNodes, gP, localPvdY.at<float>(iV, iP));
                // clang-format on
            }
        }
    }

    auto build = [](SpMat & m, std::vector<Triplet> & t)
    {
        m.setFromTriplets(t.begin(), t.end());
    };

    build(velocityMass, velocityMassT);
    build(velocityStiffness, velocityStiffnessT);

    pressureStiffnessInternalT = projectTriplets(numPressureNodes, pressureStiffnessT, internalPressureNodes);
    build(pressureInternalStiffness, pressureStiffnessInternalT);

    build(velocityPressureDiv, velocityPressureDivT);
    build(pressureVelocityDiv, pressureVelocityDivT);

    // =========================================================================================================

    const float viscosity = 0.001;

    std::vector<float> velocityXy(2 * numVelocityNodes, 0);
    std::span<float> velocityX(velocityXy.data(), numVelocityNodes);
    std::span<float> velocityY(velocityXy.data() + numVelocityNodes, numVelocityNodes);

    auto imposeDirichletVelocity = [&]()
    {
        for (const auto & dn : dirichletVx)
        {
            velocityX[dn.id] = dn.value;
        }
        for (const auto & dn : dirichletVy)
        {
            velocityY[dn.id] = dn.value;
        }
    };

    imposeDirichletVelocity();

    std::vector<float> localVx(elSizeV), localVy(elSizeV);
    cv::Mat localConvectionMatrix;
    auto assembleConvection = [&]()
    {
        convectionT.clear();

        for (int i = 0; i < nElem; i++)
        {
            const auto t = velocityMesh.elementTransforms[i];

            velocityMesh.getElement(i, idsV.data(), ptsV.data());
            for (int j = 0; j < elSizeV; j++)
            {
                const int k = idsV[j];
                localVx[j] = velocityX[k];
                localVy[j] = velocityY[k];
            }

            velocityIntegrator.integrateLocalSelfConvectionMatrix(t, localVx.data(), localVy.data(), localConvectionMatrix);
            assert(elSizeV == localConvectionMatrix.cols);
            assert(elSizeV == localConvectionMatrix.rows);

            // Accumulate
            for (int r = 0; r < elSizeV; r++)
            {
                const int i = idsV[r];
                for (int c = 0; c < elSizeV; c++)
                {
                    const int j = idsV[c];
                    const SolType val = localConvectionMatrix.at<float>(r, c);
                    const Triplet triplet(i, j, val);
                    convectionT.push_back(triplet);
                }
            }
        }

        convection.setFromTriplets(convectionT.begin(), convectionT.end());
    };

    const int numTimeSteps = std::ceil(maxT / timeStep0);
    const float tau = maxT / numTimeSteps;
    Solution result;
    result.steps.resize(numTimeSteps + 1);

    Eigen::SimplicialLDLT<SpMat> velocityMassSolver(velocityMass);
    Eigen::SimplicialLDLT<SpMat> pressureStiffnessSolver(pressureInternalStiffness);

    mesh::Interpolator velocityInterp(velocityMesh, 0.05);
    mesh::Interpolator pressureInterp(pressureMesh, 0.05);
    std::vector<cv::Scalar> colorScale{cv::Scalar(128, 0, 0), cv::Scalar(0, 0, 128), cv::Scalar(0, 255, 255)};
    const float imgScale = 1500;

    auto drawVelocity = [&](const std::string & prefix, const std::vector<float> & vx, const std::vector<float> & vy, const std::string & tag)
    {
        const float eps = 1e-2;
        auto [minX, maxX] = std::minmax_element(vx.begin(), vx.end());
        auto [minY, maxY] = std::minmax_element(vy.begin(), vy.end());

        std::cout << std::format("{}: x = [{}, {}], y = [{}, {}]\n", tag, *minX, *maxX, *minY, *maxY);

        mesh::SimpleColorScale sccX(*minX, *maxX + eps, colorScale);
        mesh::SimpleColorScale sccY(*minY, *maxY + eps, colorScale);

        velocityInterp.setValues(vx);
        auto dbgImgX = mesh::drawValues(velocityInterp, sccX, imgScale);
        velocityInterp.setValues(vy);
        auto dbgImgY = mesh::drawValues(velocityInterp, sccY, imgScale);

        cv::imwrite(prefix + "_x.png", dbgImgX);
        cv::imwrite(prefix + "_y.png", dbgImgY);
    };

    auto drawPressure = [&](const std::string & prefix, const std::vector<float> & p, const std::string & tag)
    {
        const float eps = 1e-2;
        auto [minP, maxP] = std::minmax_element(p.begin(), p.end());
        std::cout << std::format("{}: [{}, {}]\n", tag, *minP, *maxP);

        mesh::SimpleColorScale scc(*minP, *maxP + eps, colorScale);

        pressureInterp.setValues(p);
        auto dbgImgP = mesh::drawValues(pressureInterp, scc, imgScale);

        cv::imwrite(prefix + ".png", dbgImgP);
    };

    if (false)
    {
        std::vector<float> dbgVelocityX(numVelocityNodes), dbgVelocityY(numVelocityNodes);
        for (int i = 0; i < numVelocityNodes; i++)
        {
            dbgVelocityX[i] = velocityXy[i];
            dbgVelocityY[i] = velocityXy[i + numVelocityNodes];
        }
        drawVelocity("velocity_initial", dbgVelocityX, dbgVelocityY, "INITIAL");
    }

    for (int iT = 0; iT <= numTimeSteps; iT++)
    {
        const float currTime = iT * tau;
        std::cout << std::format("{} / {}: time = {}\n", iT, numTimeSteps, currTime);
        result.steps[iT].time = currTime;

        // 1) Find the tentative velocity
        // The original equation is u_t = u_i + tau(-u_i . nabla(u_i) + viscosity * delta(u_i))
        // Solve for the "acceleration" a: (a, v) = (u_i . nabla(u_i), v) + viscosity * (nabla(u_i), nabla(v))
        // Then calculate u_t = u_i - tau * a
        // The system for a is [M 0; 0 M] [a_x; a_y] = [A 0; 0 A] [q_x; q_y], where A = viscosity * M1 + C
        // Since the system is effectively the same for a_x and a_y, we can solve M [a_x a_y] = A [q_x q_y]
        assembleConvection();
        auto A = viscosity * velocityStiffness + convection;
        Eigen::Matrix<SolType, Eigen::Dynamic, 2> velocityMatrix(numVelocityNodes, 2);
        Vector tentativeVelocityXy(2 * numVelocityNodes);
        for (int i = 0; i < numVelocityNodes; i++)
        {
            velocityMatrix(i, 0) = velocityX[i];
            velocityMatrix(i, 1) = velocityY[i];
        }
        auto accelRhs = A * velocityMatrix;

        // Solve for the acceleration
        Eigen::Matrix<SolType, Eigen::Dynamic, 2> accel = velocityMassSolver.solve(accelRhs);

        // Update the velocity
        for (int i = 0; i < 2 * numVelocityNodes; i++)
        {
            tentativeVelocityXy(i) = velocityXy[i];
        }
        // Process only the internal nodes - the velocity at the start already has the correct BC's
        for (const int i : internalVelocityNodes)
        {
            // clang-format off
            tentativeVelocityXy(i)                    -= tau * accel(i, 0);
            tentativeVelocityXy(i + numVelocityNodes) -= tau * accel(i, 1);
            // clang-format on
        }

        if (false)
        {
            std::vector<float> dbgVelocityX(numVelocityNodes), dbgVelocityY(numVelocityNodes);
            for (int i = 0; i < numVelocityNodes; i++)
            {
                dbgVelocityX[i] = tentativeVelocityXy(i);
                dbgVelocityY[i] = tentativeVelocityXy(i + numVelocityNodes);
            }
            drawVelocity(std::format("velocity_{}_tentative", iT), dbgVelocityX, dbgVelocityY, "TENTATIVE");
        }

        // 2) Find the pressure: delta(p) = nabla . u_* / tau
        // Calculate the divergence of the tentative velocity
        Vector tentativeVelDiv = velocityPressureDiv * tentativeVelocityXy;
        const float invTau = -1.0f / tau;
        tentativeVelDiv *= invTau;
        assert(tentativeVelDiv.rows() == numPressureNodes);
        assert(tentativeVelDiv.cols() == 1);
        if (false)
        {
            std::vector<float> dbg(numPressureNodes);
            for (int i = 0; i < numPressureNodes; i++)
            {
                dbg[i] = tentativeVelDiv(i);
            }
            drawPressure(std::format("pressure_div_{}", iT), dbg, "DIVERGENCE");
        }

        Vector pressureRhs(numInternalPressureNodes);
        for (int i = 0; i < numInternalPressureNodes; i++)
        {
            const int j = internalPressureNodes[i];
            pressureRhs[i] = tentativeVelDiv[j];
        }

        Vector pressureInt = pressureStiffnessSolver.solve(pressureRhs);
        assert(pressureInt.rows() == numInternalPressureNodes);
        assert(pressureInt.cols() == 1);

        Vector pressure(numPressureNodes);
        pressure.setZero();
        for (int i = 0; i < numInternalPressureNodes; i++)
        {
            const int j = internalPressureNodes[i];
            pressure[j] = pressureInt[i];
        }
        if (false)
        {
            std::vector<float> dbg(numPressureNodes);
            for (int i = 0; i < numPressureNodes; i++)
            {
                dbg[i] = pressure(i);
            }
            drawPressure(std::format("pressure_{}", iT), dbg, "PRESSURE");
        }

        // Copy pressure to output
        auto & outP = result.steps[iT].pressure;
        outP.resize(numPressureNodes);
        for (int i = 0; i < numPressureNodes; i++)
        {
            outP[i] = pressure[i];
        }

        // 3) Find the final velocity by updating the tentative
        // (u_{i+1} - u_*) / tau = -nabla(p) <=> a = nabla(p) <=> (a, v) = (nabla(p), v)
        // Then update: u_{i+1} = u_* + tau * a
        auto nablaPXy = pressureVelocityDiv * pressure;
        Eigen::Matrix<SolType, Eigen::Dynamic, 2> nablaP(numVelocityNodes, 2);
        for (int i = 0; i < numVelocityNodes; i++)
        {
            nablaP(i, 0) = nablaPXy(i);
            nablaP(i, 1) = nablaPXy(i + numVelocityNodes);
        }

        Eigen::Matrix<SolType, Eigen::Dynamic, 2> accelFinal = velocityMassSolver.solve(nablaP);
        for (int i = 0; i < numVelocityNodes; i++)
        {
            // clang-format off
            velocityX[i] = tentativeVelocityXy(i)                    - tau * accelFinal(i, 0);
            velocityY[i] = tentativeVelocityXy(i + numVelocityNodes) - tau * accelFinal(i, 1);
            // clang-format on
        }
        imposeDirichletVelocity();
        result.steps[iT].velocity = velocityXy;

        if (false)
        {
            std::vector<float> dbgVelocityX(numVelocityNodes), dbgVelocityY(numVelocityNodes);
            for (int i = 0; i < numVelocityNodes; i++)
            {
                dbgVelocityX[i] = velocityXy[i];
                dbgVelocityY[i] = velocityXy[i + numVelocityNodes];
            }
            drawVelocity(std::format("velocity_{}", iT), dbgVelocityX, dbgVelocityY, "FINAL");
        }
    }

    return result;
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./NavierStokes <msh file>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string meshFileName = argv[1];
    auto triMesh = mesh::parseTriangleGmsh(meshFileName);

    const auto velocityElement = el::createElement(el::Type::P2);
    const auto pressureElement = el::createElement(el::Type::P1);

    auto velocityMesh = mesh::createMesh(triMesh, *velocityElement);
    auto pressureMesh = mesh::createMesh(triMesh, *pressureElement);

    if (true)
    {
        const float scale = 3500;
        cv::imwrite("velocity_mesh.png", mesh::drawMesh(velocityMesh, scale));
        cv::imwrite("pressure_mesh.png", mesh::drawMesh(pressureMesh, scale));
    }

    const float tau = 1e-4;
    const float maxT = 1;
    auto sol = solveNsChorinEigen(velocityMesh, pressureMesh, tau, maxT);

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

    mesh::TriangleLookup lookup(velocityMesh, 0.05);
    const float velocityStep = 0.025;
    const float velocityScale = 0.05 / 20;
    for (int i = 0; i < sol.steps.size(); i++)
    {
        const auto & s = sol.steps[i];
        const cv::Mat img = mesh::drawCfd(lookup, pressureColorScale, 800,
                                          velocityScale, velocityStep,
                                          velocityMesh, pressureMesh,
                                          s.velocity, s.pressure);
        const std::string outFname = std::format("out_{}.png", i);
        std::cout << outFname << "\n";
        cv::imwrite(outFname, img);
    }

    return 0;
}
