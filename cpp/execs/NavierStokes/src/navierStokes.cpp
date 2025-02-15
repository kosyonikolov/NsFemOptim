#include <cassert>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>

#include <Eigen/Sparse>

#include <element/affineTransform.h>
#include <element/factory.h>
#include <element/triangleIntegrator.h>

#include <mesh/colorScale.h>
#include <mesh/concreteMesh.h>
#include <mesh/drawMesh.h>
#include <mesh/gmsh.h>
#include <mesh/interpolator.h>
#include <mesh/io.h>
#include <mesh/triangleLookup.h>

#include <linalg/csrMatrix.h>
#include <linalg/eigen.h>
#include <linalg/gaussSeidel.h>

#include <utils/stopwatch.h>

#include <NavierStokes/dfgCondtions.h>
#include <NavierStokes/nsConfig.h>

using SolType = float;
using SpMat = Eigen::SparseMatrix<SolType, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<SolType>;
using Vector = Eigen::Vector<SolType, Eigen::Dynamic>;

struct FastConvection
{
    SpMat convection;
    Eigen::SparseMatrix<SolType, Eigen::RowMajor> integration;
    Vector values;
    Vector velocity; // x, then y

    FastConvection(const mesh::ConcreteMesh & velocityMesh, el::TriangleIntegrator & integrator)
    {
        const int nNodes = velocityMesh.nodes.size();
        const int nElems = velocityMesh.numElements;
        const int elSize = velocityMesh.getElementSize();

        convection = SpMat(nNodes, nNodes);
        
        // Assemble convection with fake data to create the sparse pattern
        std::vector<int> ids(elSize);
        std::vector<Triplet> fakeTriplets;
        for (int i = 0; i < nElems; i++)
        {
            velocityMesh.getElement(i, ids.data(), 0);
            for (int r = 0; r < elSize; r++)
            {
                const int globalRow = ids[r];
                for (int c = 0; c < elSize; c++)
                {
                    const int globalCol = ids[c];
                    fakeTriplets.emplace_back(globalRow, globalCol, 1);
                }
            }
        }

        convection.setFromTriplets(fakeTriplets.begin(), fakeTriplets.end());
        assert(convection.isCompressed());
        const int nnz = convection.nonZeros();
        auto pVal = convection.valuePtr();

        // Construct the integration matrix
        // It has size E x 2N, where E = nnz (nonzero entries in convection)
        integration = SpMat(nnz, 2 * nNodes);
        std::vector<float> localVx(elSize, 0);
        std::vector<float> localVy(elSize, 0);
        Eigen::Matrix<int, Eigen::Dynamic, Eigen::Dynamic> rowIndices(elSize, elSize);
        cv::Mat localConvection;
        std::vector<Triplet> integrationTriplets;
        for (int i = 0; i < nElems; i++)
        {
            velocityMesh.getElement(i, ids.data(), 0);
            // The ids determine the E-idx (row idx)
            // Calculate them once
            for (int r = 0; r < elSize; r++)
            {
                const int globalRow = ids[r];
                for (int c = 0; c < elSize; c++)
                {
                    const int globalCol = ids[c];
                    auto & ref = convection.coeffRef(globalRow, globalCol);
                    const int idx = &ref - pVal;
                    assert(idx >= 0 && idx < nnz);
                    rowIndices(r, c) = idx;
                }
            }

            // Set each velocity component to 1 and calculate the contribution
            for (int iVxy = 0; iVxy < 2 * elSize; iVxy++)
            {
                const bool isVy = iVxy >= elSize;
                std::vector<float> & localVelocity = isVy ? localVy : localVx;
                const int localIdx = isVy ? iVxy - elSize : iVxy;
                localVelocity[localIdx] = 1;
                integrator.integrateLocalSelfConvectionMatrix(velocityMesh.elementTransforms[i], localVx.data(), localVy.data(), localConvection);
                localVelocity[localIdx] = 0;

                // Accumulate to integration matrix
                const int globalNodeIdx = ids[localIdx];
                const int colIdx = isVy ? globalNodeIdx + nNodes : globalNodeIdx;
                for (int r = 0; r < elSize; r++)
                {
                    for (int c = 0; c < elSize; c++)
                    {
                        const int rowIdx = rowIndices(r, c);
                        const float val = localConvection.at<float>(r, c);
                        integrationTriplets.emplace_back(rowIdx, colIdx, val);
                    }
                }
            }
        }
        integration.setFromTriplets(integrationTriplets.begin(), integrationTriplets.end());

        velocity = Vector(2 * nNodes);
        values = Vector(nnz);
    }

    void update(const std::vector<float> & velocityXy)
    {
        const int n = velocity.rows();
        assert(n == velocityXy.size());
        for (int i = 0; i < n; i++)
        {
            velocity[i] = velocityXy[i];
        }

        values = integration * velocity;
        const int nnz = convection.nonZeros();
        assert(values.rows() == nnz);
        std::copy_n(values.data(), nnz, convection.valuePtr());
    }
};

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
                            const DfgConditions & cond, const float timeStep0, const float maxT)
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
    const auto dirichletVx = extractDirichletNodes(velocityMesh, velocityBorderIds, calcDirichletVx);
    const auto dirichletVy = extractDirichletNodes(velocityMesh, velocityBorderIds, dirichletZero);
    assert(dirichletVx.size() == dirichletVy.size());
    const auto internalVelocityNodes = extractInternalNodes(numVelocityNodes, dirichletVx);

    // Pressure
    const std::vector<int> pressureBorderIds = {idRight};
    const auto dirichletPressure = extractDirichletNodes(pressureMesh, pressureBorderIds, dirichletZero);
    const auto internalPressureNodes = extractInternalNodes(numPressureNodes, dirichletPressure);
    const int numInternalPressureNodes = internalPressureNodes.size();
    // =========================================================================================================

    // =========================================== Assemble matrices ===========================================
    std::vector<Triplet> velocityMassT, velocityStiffnessT;
    std::vector<Triplet> pressureStiffnessT, pressureStiffnessInternalT;

    // X on top, Y on bottom
    std::vector<Triplet> velocityPressureDivT;
    std::vector<Triplet> pressureVelocityDivT;

    SpMat velocityMass(numVelocityNodes, numVelocityNodes);
    SpMat velocityStiffness(numVelocityNodes, numVelocityNodes);
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

    const float viscosity = cond.viscosity;

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

    FastConvection fastConvection(velocityMesh, velocityIntegrator);

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

    SpMat A0 = viscosity * velocityStiffness;
    SpMat A;

    auto velocityMassCsr = linalg::csrFromEigen(velocityMass);
    Eigen::Matrix<SolType, Eigen::Dynamic, 2> accelRhs;
    // Interleaved XY
    std::vector<float> tentRhs(2 * velocityMassCsr.rows);
    std::vector<float> tentAcc(2 * velocityMassCsr.rows);

    for (int iT = 0; iT <= numTimeSteps; iT++)
    {
        u::Stopwatch bigSw;
        u::Stopwatch sw;

        const float currTime = iT * tau;
        std::cout << std::format("{} / {}: time = {}\n", iT, numTimeSteps, currTime);
        result.steps[iT].time = currTime;

        // 1) Find the tentative velocity
        // The original equation is u_t = u_i + tau(-u_i . nabla(u_i) + viscosity * delta(u_i))
        // Solve for the "acceleration" a: (a, v) = (u_i . nabla(u_i), v) + viscosity * (nabla(u_i), nabla(v))
        // Then calculate u_t = u_i - tau * a
        // The system for a is [M 0; 0 M] [a_x; a_y] = [A 0; 0 A] [q_x; q_y], where A = viscosity * M1 + C
        // Since the system is effectively the same for a_x and a_y, we can solve M [a_x a_y] = A [q_x q_y]
        fastConvection.update(velocityXy);
        const auto tAsmConvection = sw.millis(true);

        A = A0 + fastConvection.convection;
        Eigen::Matrix<SolType, Eigen::Dynamic, 2> velocityMatrix(numVelocityNodes, 2);
        Vector tentativeVelocityXy(2 * numVelocityNodes);
        for (int i = 0; i < numVelocityNodes; i++)
        {
            velocityMatrix(i, 0) = velocityX[i];
            velocityMatrix(i, 1) = velocityY[i];
        }
        Eigen::Matrix<SolType, Eigen::Dynamic, 2> accelRhs = A * velocityMatrix;

        // Solve for the acceleration
        for (int i = 0; i < numVelocityNodes; i++)
        {
            tentRhs[2 * i + 0] = accelRhs(i, 0);
            tentRhs[2 * i + 1] = accelRhs(i, 1);
        }
        constexpr double eps = 1e-6;
        linalg::gaussSeidel2ch(velocityMassCsr, tentAcc, tentRhs, 100, eps);

        // Update the velocity
        for (int i = 0; i < 2 * numVelocityNodes; i++)
        {
            tentativeVelocityXy(i) = velocityXy[i];
        }
        // Process only the internal nodes - the velocity at the start already has the correct BC's
        for (const int i : internalVelocityNodes)
        {
            // clang-format off
            tentativeVelocityXy(i)                    -= tau * tentAcc[2 * i + 0];
            tentativeVelocityXy(i + numVelocityNodes) -= tau * tentAcc[2 * i + 1];
            // clang-format on
        }
        const auto tTentative = sw.millis(true);

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

        const auto tPressure = sw.millis(true);

        // 3) Find the final velocity by updating the tentative
        // (u_{i+1} - u_*) / tau = -nabla(p) <=> a = nabla(p) <=> (a, v) = (nabla(p), v)
        // Then update: u_{i+1} = u_* + tau * a
        Vector nablaPXy = pressureVelocityDiv * pressure;
        const auto tCalcNablaPxy = sw.millis(true);
        Eigen::Matrix<SolType, Eigen::Dynamic, 2> nablaP(numVelocityNodes, 2);
        for (int i = 0; i < numVelocityNodes; i++)
        {
            nablaP(i, 0) = nablaPXy(i);
            nablaP(i, 1) = nablaPXy(i + numVelocityNodes);
        }
        const auto tCalcNablaP = sw.millis(true);

        Eigen::Matrix<SolType, Eigen::Dynamic, 2> accelFinal = velocityMassSolver.solve(nablaP);
        const auto tSolveFinal = sw.millis(true);

        for (int i = 0; i < numVelocityNodes; i++)
        {
            // clang-format off
            velocityX[i] = tentativeVelocityXy(i)                    - tau * accelFinal(i, 0);
            velocityY[i] = tentativeVelocityXy(i + numVelocityNodes) - tau * accelFinal(i, 1);
            // clang-format on
        }

        const auto tUpdateFinal = sw.millis();

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

        const auto tIter = bigSw.millis();
        std::cout << std::format("Total time = {}, assemble convection = {}, tentative = {}, pressure = {}, nablaPxy = {}, nablaP = {}, solveFinal = {}, updateFinal = {}\n",
                                 tIter, tAsmConvection, tTentative, tPressure, tCalcNablaPxy, tCalcNablaP, tSolveFinal, tUpdateFinal);
    }

    return result;
}

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

    DfgConditions cond;
    cond.viscosity = cfg.viscosity;
    cond.peakVelocity = cfg.peakVelocity;

    const float tau = cfg.tau;
    const float maxT = cfg.maxT;
    auto sol = solveNsChorinEigen(velocityMesh, pressureMesh, cond, tau, maxT);

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
