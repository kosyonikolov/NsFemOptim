#include <NavierStokes/chorinEigen.h>

#include <iostream>
#include <stdexcept>

#include <Eigen/Sparse>

#include <element/affineTransform.h>
#include <element/triangleIntegrator.h>

#include <mesh/colorScale.h>
#include <mesh/concreteMesh.h>
#include <mesh/drawMesh.h>
#include <mesh/gmsh.h>
#include <mesh/interpolator.h>

#include <linalg/csrMatrix.h>
#include <linalg/eigen.h>
#include <linalg/gaussSeidel.h>
#include <linalg/io.h>

#include <utils/stopwatch.h>

#include <fem/chorinMatrices.h>
#include <fem/dirichletNode.h>
#include <fem/fastConvection.h>

#include <NavierStokes/borders.h>
#include <NavierStokes/dfgCondtions.h>
#include <NavierStokes/nsConfig.h>
#include <NavierStokes/solution.h>

using SolType = float;
using SpMat = Eigen::SparseMatrix<SolType, Eigen::RowMajor>;
using Triplet = Eigen::Triplet<SolType>;
using Vector = Eigen::Vector<SolType, Eigen::Dynamic>;

void downloadEigen(const Vector & src, std::vector<float> & dst)
{
    const int n = src.rows();
    dst.resize(n);
    for (int i = 0; i < n; i++)
    {
        dst[i] = src[i];
    }
}

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
    const auto dirichletVx = extractDirichletNodes(velocityMesh, velocityBorderIds, calcDirichletVx);
    const auto dirichletVy = extractDirichletNodes(velocityMesh, velocityBorderIds, dirichletZero);
    assert(dirichletVx.size() == dirichletVy.size());
    const auto internalVelocityNodes = extractInternalNodes(numVelocityNodes, dirichletVx);

    // Pressure
    const std::vector<int> pressureBorderIds = {idRight};
    const auto dirichletPressure = extractDirichletNodes(pressureMesh, pressureBorderIds, dirichletZero);
    const auto internalPressureNodes = extractInternalNodes(numPressureNodes, dirichletPressure);
    const int numInternalPressureNodes = internalPressureNodes.size();
    std::cout << "Done\n";
    // =========================================================================================================

    // =========================================== Assemble matrices ===========================================
    std::cout << "Assembling matrices... ";
    std::cout.flush();

    const int integrationDegree = 4;
    el::TriangleIntegrator velocityIntegrator(velocityMesh.baseElement, integrationDegree, pressureMesh.baseElement);

    u::Stopwatch bigSw;
    u::Stopwatch smallSw;
    auto chorinBuilders = fem::buildChorinMatrices<SolType>(velocityMesh, pressureMesh, integrationDegree);
    const auto tBuilders = smallSw.millis(true);
    auto velocityMassCsr = chorinBuilders.velocityMass.buildCsr();
    auto velocityStiffnessCsr = chorinBuilders.velocityStiffness.buildCsr();
    auto pressureStiffnessCsr = chorinBuilders.pressureStiffness.buildCsr();
    auto velocityPressureDivCsr = chorinBuilders.velocityPressureDiv.buildCsr();
    auto pressureVelocityDivCsr = chorinBuilders.pressureVelocityDiv.buildCsr();
    const auto tCsrs = smallSw.millis(true);

    auto pressureStiffnessInternalCsr = pressureStiffnessCsr.slice(internalPressureNodes, internalPressureNodes);
    const auto tSlice = smallSw.millis(true);

    // Convert to Eigen
    auto velocityMass = linalg::eigenFromCsr(velocityMassCsr);
    auto velocityStiffness = linalg::eigenFromCsr(velocityStiffnessCsr);
    auto pressureInternalStiffness = linalg::eigenFromCsr(pressureStiffnessInternalCsr);
    auto velocityPressureDiv = linalg::eigenFromCsr(velocityPressureDivCsr);
    auto pressureVelocityDiv = linalg::eigenFromCsr(pressureVelocityDivCsr);
    const auto tCvtEigen = smallSw.millis();
    const auto tTotal = bigSw.millis();

    std::cout << std::format("CSR assembly (ms): total = {}, builders = {}, CSRs = {}, slice = {}, cvtEigen = {}\n",
                             tTotal, tBuilders, tCsrs, tSlice, tCvtEigen);

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

    std::cout << "Setting up FastConvection... ";
    fem::FastConvection fastConvection(velocityMesh, velocityIntegrator);
    std::cout << "Done\n";

    const int numTimeSteps = std::ceil(maxT / timeStep0);
    const float tau = maxT / numTimeSteps;
    Solution result;
    result.steps.resize(numTimeSteps + 1);

    Eigen::SimplicialLDLT<SpMat> velocityMassSolver(velocityMass);
    Eigen::SimplicialLDLT<SpMat> pressureStiffnessSolver(pressureInternalStiffness);
    if (false)
    {
        auto dumpM = linalg::csrFromEigen(pressureInternalStiffness);
        linalg::write("dump/pressure_mat.bin", dumpM);
    }

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

    // linalg::write("dump/velocityMass.bin", velocityMassCsr);

    Eigen::Matrix<SolType, Eigen::Dynamic, 2> accelRhs;
    // Interleaved XY
    std::vector<float> tentRhs(2 * velocityMassCsr.rows);
    std::vector<float> tentAcc(2 * velocityMassCsr.rows);

    // ======= Debug dumps =======
    const std::string dumpDir = "dumps_eigen";
    const bool dbgDumps = false;
    std::vector<float> dbgFinalAccRhs(2 * numVelocityNodes);
    std::vector<float> dbgVelocityXy;
    std::vector<float> dbgPressureRhs;
    std::vector<float> dbgInternalP;
    std::vector<float> dbgFullP;

    if (dbgDumps)
    {
        linalg::write(dumpDir + "/velocityMass.bin", velocityMassCsr);
    }

    std::cout << "Solving...\n";
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

        if (dbgDumps)
        {
            linalg::write(std::format("{}/{}_tentativeRhs.bin", dumpDir, iT), tentRhs);
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

        if (dbgDumps)
        {
            downloadEigen(tentativeVelocityXy, dbgVelocityXy);
            linalg::write(std::format("{}/{}_tentativeVxy.bin", dumpDir, iT), dbgVelocityXy);
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

        if (dbgDumps)
        {
            downloadEigen(pressureRhs, dbgPressureRhs);
            linalg::write(std::format("{}/{}_pressureRhs.bin", dumpDir, iT), dbgPressureRhs);
        }

        Vector pressureInt = pressureStiffnessSolver.solve(pressureRhs);
        assert(pressureInt.rows() == numInternalPressureNodes);
        assert(pressureInt.cols() == 1);
        if (false)
        {
            const int n = pressureInt.rows();
            std::vector<float> rhs(n);
            std::vector<float> sol(n);
            for (int i = 0; i < n; i++)
            {
                rhs[i] = pressureRhs[i];
                sol[i] = pressureInt[i];
            }
            linalg::write(std::format("dump/pressure_rhs_{}.bin", iT), rhs);
            linalg::write(std::format("dump/pressure_sol_{}.bin", iT), sol);
        }

        Vector pressure(numPressureNodes);
        pressure.setZero();
        for (int i = 0; i < numInternalPressureNodes; i++)
        {
            const int j = internalPressureNodes[i];
            pressure[j] = pressureInt[i];
        }

        if (dbgDumps)
        {
            downloadEigen(pressureInt, dbgInternalP);
            downloadEigen(pressure, dbgFullP);
            linalg::write(std::format("{}/{}_internalP.bin", dumpDir, iT), dbgInternalP);
            linalg::write(std::format("{}/{}_fullP.bin", dumpDir, iT), dbgFullP);
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

        if (dbgDumps)
        {
            for (int i = 0; i < numVelocityNodes; i++)
            {
                dbgFinalAccRhs[2 * i + 0] = nablaP(i, 0);
                dbgFinalAccRhs[2 * i + 1] = nablaP(i, 1);
            }
            linalg::write(std::format("{}/{}_accelFinalRhs.bin", dumpDir, iT), dbgFinalAccRhs);
        }

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
