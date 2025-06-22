#include <NavierStokes/chorinCuda.h>

#include <cassert>
#include <format>
#include <stdexcept>

#include <cu/blas.h>
#include <cu/conjGradF.h>
#include <cu/csrF.h>
#include <cu/sparse.h>
#include <cu/spmv.h>

#include <linalg/io.h>

#include <utils/stopwatch.h>

#include <NavierStokes/buildContext.h>

struct DirichletVelocity
{
    cu::Sparse & lib;

    cu::vec<float> & velocityXy; // X, then Y
    cu::vec<int> ids;            // size = n
    cu::vec<float> values;       // size = n

    cusparseDnVecDescr_t denseVec;  // velocityXy
    cusparseSpVecDescr_t sparseVec; // ids, values

    DirichletVelocity(cu::Sparse & sparseHandle,
                      cu::vec<float> & velocityXy,
                      const std::vector<fem::DirichletNode> & x,
                      const std::vector<fem::DirichletNode> & y)
        : lib(sparseHandle), velocityXy(velocityXy)
    {
        const int numNodes = velocityXy.size();
        assert(numNodes % 2 == 0);
        const int numNodesChannel = numNodes / 2;

        const int nnz = x.size() + y.size();
        std::vector<int> cpuIds(nnz);
        std::vector<float> cpuVals(nnz);
        int i = 0;
        for (int j = 0; j < x.size(); j++, i++)
        {
            cpuIds[i] = x[j].id;
            cpuVals[i] = x[j].value;
        }
        for (int j = 0; j < y.size(); j++, i++)
        {
            cpuIds[i] = y[j].id + numNodesChannel;
            cpuVals[i] = y[j].value;
        }

        ids.overwriteUpload(cpuIds);
        values.overwriteUpload(cpuVals);

        auto rc = cusparseCreateSpVec(&sparseVec, numNodes, nnz, ids.get(), values.get(),
                                      cusparseIndexType_t::CUSPARSE_INDEX_32I,
                                      cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                                      cudaDataType::CUDA_R_32F);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create cusparse sparse vector: {}", cusparseGetErrorName(rc)));
        }

        denseVec = velocityXy.getCuSparseDescriptor();
    }

    void impose()
    {
        auto rc = cusparseScatter(lib.handle(), sparseVec, denseVec);
    }
};

struct PressureSolver
{
    cu::Sparse & lib;
    cu::csrF & pressureStiffnessInternal;
    cu::ConjugateGradientF cg;

    int numAll;
    int numInternal;

    float cgTarget = 1e-6;
    int cgMaxIters = 1000;

    // Input/output buffer
    // Before pressure is calculated, this is tentativeVelDiv
    // After it is calculated, this is the pressure
    cu::vec<float> dense;
    cu::vec<int> internalIds;
    cu::vec<float> rhs;
    cu::vec<float> internalPressure;

    cusparseSpVecDescr_t sparseInput;  // values = rhs
    cusparseSpVecDescr_t sparseOutput; // values = internalPressure

    float lastMse;

    PressureSolver(cu::Sparse & lib, cu::csrF & m1,
                   const int numPressureNodes,
                   const std::vector<int> & internalPressureIds)
        : lib(lib), pressureStiffnessInternal(m1), cg(m1),
          dense(numPressureNodes),
          internalIds(internalPressureIds),
          rhs(internalPressureIds.size()),
          internalPressure(internalPressureIds.size())
    {
        assert(m1.cols == m1.rows);
        assert(m1.cols == internalPressureIds.size());
        numAll = numPressureNodes;
        numInternal = internalPressureIds.size();
        assert(numInternal > 0 && numInternal <= numAll);

        auto rc = cusparseCreateSpVec(&sparseInput, numAll, numInternal,
                                      internalIds.get(),
                                      rhs.get(),
                                      cusparseIndexType_t::CUSPARSE_INDEX_32I,
                                      cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                                      cudaDataType::CUDA_R_32F);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseCreateSpVec failed: {}", cusparseGetErrorName(rc)));
        }

        rc = cusparseCreateSpVec(&sparseOutput, numAll, numInternal,
                                 internalIds.get(),
                                 internalPressure.get(),
                                 cusparseIndexType_t::CUSPARSE_INDEX_32I,
                                 cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                                 cudaDataType::CUDA_R_32F);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseCreateSpVec failed: {}", cusparseGetErrorName(rc)));
        }
    }

    void update()
    {
        auto rc = cusparseGather(lib.handle(), dense.getCuSparseDescriptor(), sparseInput);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseGather failed: {}", cusparseGetErrorName(rc)));
        }

        lastMse = cg.solve(rhs, internalPressure, cgMaxIters, cgTarget);
        if (!std::isfinite(lastMse))
        {
            throw std::runtime_error("Bad CG");
        }

        // Output pressure
        dense.memsetZero();
        rc = cusparseScatter(lib.handle(), sparseOutput, dense.getCuSparseDescriptor());
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseScatter failed: {}", cusparseGetErrorName(rc)));
        }
    }
};

Solution solveNsChorinCuda(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                           const DfgConditions & cond, const float timeStep0, const float maxT)
{
    cu::Blas blas;
    cu::Sparse sparse;

    float plusOne = 1.0f;

    auto ctx = buildChorinContext(velocityMesh, pressureMesh, cond);

    // Create CUDA matrices
    cu::csrF velocityMass(ctx.velocityMass);
    cu::csrF velocityStiffnessPlusConvection(ctx.velocityStiffness);
    cu::csrF pressureStiffnessInternal(ctx.pressureStiffnessInternal);
    cu::csrF velocityPressureDiv(ctx.velocityPressureDiv);
    cu::csrF pressureVelocityDiv(ctx.pressureVelocityDiv);
    cu::csrF fastConvectionIntegration(ctx.fastConvectionIntegration);

    // Copy original stiffness matrix values
    // On each iteration we will do A = viscosity * M1 + C and store the result in velocityStiffnessPlusConvection
    cu::vec<float> origVelocityM1Vals(velocityStiffnessPlusConvection.values);
    float viscosity = cond.viscosity;
    auto blasRc = cublasSscal(blas.handle, origVelocityM1Vals.size(), &viscosity, origVelocityM1Vals.get(), 1);
    if (blasRc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::format("Failed to scale M1: {}", cublasGetStatusName(blasRc)));
    }

    cu::spmv fcSpmv(sparse.handle(), fastConvectionIntegration);
    cu::spmv aSpmv(sparse.handle(), velocityStiffnessPlusConvection);

    const int numVelocityNodes = ctx.numVelocityNodes;
    const int numPressureNodes = ctx.numPressureNodes;

    cu::vec<float> velocityXy(2 * numVelocityNodes); // X, then Y
    velocityXy.memsetZero();

    DirichletVelocity dirichletVelocity(sparse, velocityXy, ctx.dirichletVx, ctx.dirichletVy);
    dirichletVelocity.impose();
    // Create overlapping views of the X and Y velocities
    cu::vec<float> velocityX(velocityXy.get(), numVelocityNodes);
    cu::vec<float> velocityY(velocityXy.get() + numVelocityNodes, numVelocityNodes);

    // Acceleration
    cu::vec<float> accel(numVelocityNodes);
    cu::ConjugateGradientF velocityMassCg(velocityMass);
    const float cgTargetAccel = 1e-6f;
    const int cgMaxItersAccel = 1000;

    // Pressure
    cu::spmv vpdSpmv(sparse.handle(), velocityPressureDiv);
    PressureSolver pressureSolver(sparse, pressureStiffnessInternal,
                                  numPressureNodes, ctx.internalPressureNodes);
    cu::spmv pvdSpmv(sparse.handle(), pressureVelocityDiv);
    auto & nablaPXy = pvdSpmv.b;
    assert(nablaPXy.size() == 2 * numVelocityNodes);
    // Create vectors for the X and Y components of nabla
    cu::vec<float> nablaPX(nablaPXy.get(), numVelocityNodes);
    cu::vec<float> nablaPY(nablaPXy.get() + numVelocityNodes, numVelocityNodes);

    const int numTimeSteps = std::ceil(maxT / timeStep0);
    const float tau = maxT / numTimeSteps;
    const float invTau = -1.0f / tau;
    Solution result;
    result.steps.resize(numTimeSteps + 1);

    // ======= Debug dumps =======
    const std::string dumpDir = "dumps_cuda";
    const bool dbgDumps = false;
    std::vector<float> dbgVelocityXy(velocityXy.size());
    std::vector<float> dbgPressureRhs(pressureSolver.rhs.size());
    std::vector<float> dbgInternalP(ctx.internalPressureNodes.size());
    std::vector<float> dbgFullP(numPressureNodes);

    for (int iT = 0; iT <= numTimeSteps; iT++)
    {
        u::Stopwatch bigSw;
        u::Stopwatch sw;

        // Update convection
        auto & currConvection = fcSpmv.b;
        fcSpmv.compute(velocityXy, currConvection);
        // Calculate A = viscosity * M1 + convection
        auto & aValues = velocityStiffnessPlusConvection.values;
        blasRc = cublasSgeam(blas.handle, cublasOperation_t::CUBLAS_OP_N, cublasOperation_t::CUBLAS_OP_N,
                             aValues.size(), 1,
                             &plusOne, origVelocityM1Vals.get(), aValues.size(),
                             &plusOne, currConvection.get(), aValues.size(),
                             aValues.get(), aValues.size());
        if (blasRc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cublasSgeam failed: {}", cublasGetStatusName(blasRc)));
        }
        const auto tConvection = sw.millis(true);

        // =========================================================================================
        // Find tentative velocity in two steps:
        // 1) Compute accelRhsC = A * velocityC
        // 2) Solve M0 * accelC = accelRhsC
        // Repeat for C = {X, Y}

        // X
        auto & accelRhs = aSpmv.b;
        aSpmv.compute(velocityX, accelRhs);
        accel.memsetZero();
        const float mseTentX = velocityMassCg.solve(accelRhs, accel, cgMaxItersAccel, cgTargetAccel);
        if (!std::isfinite(mseTentX))
        {
            throw std::runtime_error("Bad CG");
        }
        // v* = v - tau * accel
        cu::saxpy(blas, numVelocityNodes, accel.get(), velocityX.get(), -tau);

        // Y
        aSpmv.compute(velocityY, accelRhs);
        accel.memsetZero();
        const float mseTentY = velocityMassCg.solve(accelRhs, accel, cgMaxItersAccel, cgTargetAccel);
        if (!std::isfinite(mseTentY))
        {
            throw std::runtime_error("Bad CG");
        }
        // v* = v - tau * accel
        cu::saxpy(blas, numVelocityNodes, accel.get(), velocityY.get(), -tau);

        // Reimpose BCs
        dirichletVelocity.impose();
        if (dbgDumps)
        {
            velocityXy.download(dbgVelocityXy);
            linalg::write(std::format("{}/{}_tentativeVxy.bin", dumpDir, iT), dbgVelocityXy);
        }
        const auto tTentative = sw.millis(true);
        // =========================================================================================

        // =========================================================================================
        // Find the pressure:
        // 1) Compute the RHS of the system:
        //      tentativeVelDiv = velocityPressureDiv * tentativeVelocityXy;
        //      pressureRhs = gather(tenativeVelDiv, internalPressureNodes)
        // 2) Find the internal pressure: pressureInt = pressureStiffnessSolver.solve(pressureRhs);
        // 3) Scatter the internal pressure

        // delta(p) = nabla . u_* / tau
        // Calculate the divergence of the tentative velocity
        vpdSpmv.compute(velocityXy, pressureSolver.dense);
        cu::scale(blas, pressureSolver.dense.size(), pressureSolver.dense.get(), invTau);

        pressureSolver.update();
        const float msePressure = pressureSolver.lastMse;
        if (dbgDumps)
        {
            pressureSolver.rhs.download(dbgPressureRhs);
            linalg::write(std::format("{}/{}_pressureRhs.bin", dumpDir, iT), dbgPressureRhs);
        }

        auto & pressure = pressureSolver.dense;
        assert(pressure.size() == numPressureNodes);

        if (dbgDumps)
        {
            pressureSolver.internalPressure.download(dbgInternalP);
            pressure.download(dbgFullP);
            linalg::write(std::format("{}/{}_internalP.bin", dumpDir, iT), dbgInternalP);
            linalg::write(std::format("{}/{}_fullP.bin", dumpDir, iT), dbgFullP);
        }

        const auto tPressure = sw.millis(true);

        // Copy to output
        auto & outP = result.steps[iT].pressure;
        outP.resize(numPressureNodes);
        pressure.download(outP);

        const auto tPressureDownload = sw.millis();
        // =========================================================================================

        // =========================================================================================
        // Find the final velocity by updating the tentative
        // (u_{i+1} - u_*) / tau = -nabla(p) <=>
        // <=> a = nabla(p) <=>
        // <=> (a, v) = (nabla(p), v)
        // Then update: u_{i+1} = u_* + tau * a
        // Calculate X and Y channels separately

        // nablaPXy = pressureVelocityDiv * pressure;
        pvdSpmv.compute(pressure, nablaPXy);

        // X
        accel.memsetZero();
        const float mseFinalX = velocityMassCg.solve(nablaPX, accel, cgMaxItersAccel, cgTargetAccel);
        if (!std::isfinite(mseFinalX))
        {
            throw std::runtime_error("Bad CG");
        }
        cu::saxpy(blas, numVelocityNodes, accel.get(), velocityX.get(), -tau);

        // Y
        accel.memsetZero();
        const float mseFinalY = velocityMassCg.solve(nablaPY, accel, cgMaxItersAccel, cgTargetAccel);
        if (!std::isfinite(mseFinalY))
        {
            throw std::runtime_error("Bad CG");
        }
        cu::saxpy(blas, numVelocityNodes, accel.get(), velocityY.get(), -tau);

        dirichletVelocity.impose();

        const float tFinal = sw.millis(true);

        // Copy to output
        auto & outVelocity = result.steps[iT].velocity;
        outVelocity.resize(velocityXy.size());
        velocityXy.download(outVelocity);

        const float tFinalDownload = sw.millis();
        const float tIter = bigSw.millis();
        // =========================================================================================

        std::cout << std::format("{} / {}: {} ms\n", iT, numTimeSteps, tIter);
        std::cout << std::format("\tconvection = {}, tentative = {}, pressure = {}, pressureDownload = {}, final = {}, finalDownload = {}\n",
                                 tConvection, tTentative, tPressure, tPressureDownload, tFinal, tFinalDownload);
        std::cout << std::format("\tMSEs: tentX = {}, tentY = {}, pressure = {}, finalX = {}, finalY = {}\n",
                                 mseTentX, mseTentY, msePressure, mseFinalX, mseFinalY);

    }

    return result;
}