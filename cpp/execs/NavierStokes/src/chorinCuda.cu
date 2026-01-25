#include <NavierStokes/chorinCuda.h>

#include <cassert>
#include <filesystem>
#include <format>
#include <stdexcept>

#include <cu/blas.h>
#include <cu/csrF.h>
#include <cu/event.h>
#include <cu/solvers/solverFactory.h>
#include <cu/sparse.h>
#include <cu/spmm.h>
#include <cu/spmv.h>

#include <linalg/io.h>

#include <utils/stopwatch.h>

#include <NavierStokes/buildContext.h>
#include <NavierStokes/log.h>
#include <NavierStokes/progressTracker.h>

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
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to scatter velocity: {}", cusparseGetErrorName(rc)));
        }
    }
};

struct DirichletAcceleration
{
    cu::Sparse & lib;

    cu::vec<float> &accel, &accelInt;
    cu::vec<float> &rhs, &rhsInt;

    cu::vec<int> internalIds;

    cusparseDnVecDescr_t accelDenseVec;
    cusparseSpVecDescr_t accelSparseVec;

    cusparseDnVecDescr_t rhsDenseVec;
    cusparseSpVecDescr_t rhsSparseVec;

    DirichletAcceleration(cu::Sparse & sparseHandle,
                          cu::vec<float> & accel, cu::vec<float> & accelInt,
                          cu::vec<float> & rhs, cu::vec<float> & rhsInt,
                          const std::vector<fem::DirichletNode> & x)
        : lib(sparseHandle),
          accel(accel), accelInt(accelInt),
          rhs(rhs), rhsInt(rhsInt)
    {
        const int numNodes = accel.size();
        assert(numNodes % 2 == 0);
        assert(numNodes == rhs.size());
        const int numNodesChannel = numNodes / 2;

        const int numInternal = accelInt.size();
        assert(numInternal % 2 == 0);
        assert(numInternal == rhsInt.size());
        const int numInternalChannel = numInternal / 2;

        const int numDirichlet = x.size();
        assert(numDirichlet + numInternalChannel == numNodesChannel);

        // Extract Dirichlet node IDs and sort them
        // Then compute the internal nodes = setdiff([0, N), dirichlet)
        std::vector<int> dirichletIds;
        dirichletIds.reserve(numDirichlet);
        for (const auto & node : x)
        {
            dirichletIds.push_back(node.id);
        }
        assert(dirichletIds.size() == numDirichlet);
        std::sort(dirichletIds.begin(), dirichletIds.end());

        // Single channel first - Y channel will be appended later
        std::vector<int> cpuInternalIds(numInternal);
        int intIdx = 0;
        int dIdx = 0;
        int i = 0;
        while (intIdx < numInternalChannel && dIdx < numDirichlet)
        {
            const int d = dirichletIds[dIdx];
            if (i < d)
            {
                cpuInternalIds[intIdx] = i;
                intIdx++;
                i++;
            }
            else if (i == d)
            {
                i++;
                dIdx++;
            }
            else
            {
                // Should never happen - each Dirichlet ID should be matched by a sequential ID
                throw std::invalid_argument(std::format("Encountered invalid dirichlet ID [{}], which is outside of the valid range", d));
            }
        }
        // Dirichlet nodes could run out before internal nodes
        while (intIdx < numInternalChannel)
        {
            cpuInternalIds[intIdx] = i;
            intIdx++;
            i++;
        }
        assert(intIdx == numInternalChannel);
        assert(dIdx == numDirichlet);

        // Clone the X channel to Y
        for (int i = 0; i < numInternalChannel; i++)
        {
            cpuInternalIds[i + numInternalChannel] = cpuInternalIds[i] + numNodesChannel;
        }

        // Upload
        internalIds.overwriteUpload(cpuInternalIds);

        // Acceleration
        auto rc = cusparseCreateSpVec(&accelSparseVec, numNodes, numInternal, internalIds.get(), accelInt.get(),
                                      cusparseIndexType_t::CUSPARSE_INDEX_32I,
                                      cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                                      cudaDataType::CUDA_R_32F);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create cusparse sparse vector for acceleration: {}", cusparseGetErrorName(rc)));
        }

        // Right-hand side
        rc = cusparseCreateSpVec(&rhsSparseVec, numNodes, numInternal, internalIds.get(), rhsInt.get(),
                                 cusparseIndexType_t::CUSPARSE_INDEX_32I,
                                 cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                                 cudaDataType::CUDA_R_32F);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create cusparse sparse vector for acceleration RHS: {}", cusparseGetErrorName(rc)));
        }

        accelDenseVec = accel.getCuSparseDescriptor();
        rhsDenseVec = rhs.getCuSparseDescriptor();
    }

    // Copy internal elements of RHS into rhsInt
    void sliceRhs()
    {
        auto rc = cusparseGather(lib.handle(), rhsDenseVec, rhsSparseVec);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to gather acceleration RHS: {}", cusparseGetErrorName(rc)));
        }
    }

    // Copy internal acceleartion to acceleration
    void scatterAcceleration()
    {
        auto rc = cusparseScatter(lib.handle(), accelSparseVec, accelDenseVec);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to scatter acceleration: {}", cusparseGetErrorName(rc)));
        }
    }
};

struct PressureSolver
{
    cu::Sparse & lib;

    cu::AbstractSolver & solver;

    int numAll;
    int numInternal;

    // Input/output buffer
    // Before pressure is calculated, this is tentativeVelDiv
    // After it is calculated, this is the pressure
    cu::vec<float> dense;
    cu::vec<int> internalIds;

    cusparseSpVecDescr_t sparseInput;  // values = rhs
    cusparseSpVecDescr_t sparseOutput; // values = internalPressure

    PressureSolver(cu::AbstractSolver & solver, cu::Sparse & lib,
                   const int numPressureNodes,
                   const std::vector<int> & internalPressureIds)
        : lib(lib), solver(solver),
          dense(numPressureNodes),
          internalIds(internalPressureIds)
    {
        numAll = numPressureNodes;
        numInternal = internalPressureIds.size();
        assert(numInternal > 0 && numInternal <= numAll);

        auto & rhs = solver.getRhs();
        auto & internalPressure = solver.getSol();

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

        // solver.rhs is now updated
        solver.solve();
        // solver.sol is now updated

        // Output pressure
        dense.memsetZero();
        rc = cusparseScatter(lib.handle(), sparseOutput, dense.getCuSparseDescriptor());
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseScatter failed: {}", cusparseGetErrorName(rc)));
        }
    }
};

void printDebugFileName(const std::string & fileName)
{
    std::cout << "\e[1;33m" << fileName << "\e[0m\n";
}

void solveNsChorinCuda(const mesh::ConcreteMesh & velocityMesh, const mesh::ConcreteMesh & pressureMesh,
                       const DfgConditions & cond, const float timeStep0, const float maxT,
                       const ChorinCudaConfig & cfg, AbstractOutputHandler & outputHandler)
{
    cu::Blas blas;
    cu::Sparse sparse;

    Log log("chorin_cuda.csv");

    float plusOne = 1.0f;

    auto ctx = buildChorinContext(velocityMesh, pressureMesh, cond);

    // ======= Debug dumps =======
    const auto dumpCfg = cfg.dbgDumps;
    const bool dbgDumps = dumpCfg.enabled;
    const std::string dumpDir = dumpCfg.dir;
    if (dbgDumps)
    {
        std::filesystem::create_directories(dumpDir);

        // Dump matrices
        auto dump = [dumpDir](const std::string & name, const linalg::CsrMatrix<float> & m)
        {
            const std::string outFname = dumpDir + "/" + name + ".bin";
            printDebugFileName(outFname);
            linalg::write(outFname, m);
        };

#define DUMP(x) dump(#x, ctx.x)
        DUMP(velocityMassInternal);
        DUMP(velocityStiffness);
        DUMP(pressureStiffness);
        DUMP(pressureStiffnessInternal);
        DUMP(velocityPressureDiv);
        DUMP(pressureVelocityDiv);
        DUMP(fastConvectionIntegration);
#undef DUMP
    }

    // Create CUDA matrices
    cu::csrF velocityMassInternal(ctx.velocityMassInternal);
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
    cu::spmm aSpmm(sparse.handle(), velocityStiffnessPlusConvection, 2);

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
    // Only solve for internal nodes, hence the two vectors for the left and right hand side
    cu::vec<float> accel(2 * numVelocityNodes);
    cu::vec<float> accelRhs(2 * numVelocityNodes);
    const auto & vSolverCfg = cfg.velocitySolver;
    auto velocitySolver = cu::createSolver(vSolverCfg.method, 2,
                                           ctx.velocityMassInternal, vSolverCfg.maxIterations,
                                           vSolverCfg.targetMse, vSolverCfg.mseCheckInterval);
    auto & accelRhsInt = velocitySolver->getRhs();
    auto & accelInt = velocitySolver->getSol();

    assert(ctx.dirichletVx.size() == ctx.dirichletVy.size());
    DirichletAcceleration dirichletAcceleration(sparse, accel, accelInt, accelRhs, accelRhsInt, ctx.dirichletVx);

    // Default state of accelration vector
    // Dirichlet nodes will remain zero throughout the run
    accel.memsetZero();

    // Pressure
    cu::spmv vpdSpmv(sparse.handle(), velocityPressureDiv);
    const auto & pSolverCfg = cfg.pressureSolver;
    auto pressureSolverCore = cu::createSolver(pSolverCfg.method, 1,
                                               ctx.pressureStiffnessInternal, pSolverCfg.maxIterations,
                                               pSolverCfg.targetMse, pSolverCfg.mseCheckInterval);
    PressureSolver pressureSolver(*pressureSolverCore, sparse, numPressureNodes, ctx.internalPressureNodes);

    cu::spmv pvdSpmv(sparse.handle(), pressureVelocityDiv);
    auto & nablaPXy = pvdSpmv.b;
    assert(nablaPXy.size() == 2 * numVelocityNodes);
    // Create vectors for the X and Y components of nabla
    cu::vec<float> nablaPX(nablaPXy.get(), numVelocityNodes);
    cu::vec<float> nablaPY(nablaPXy.get() + numVelocityNodes, numVelocityNodes);

    const int numTimeSteps = std::ceil(maxT / timeStep0);
    const float tau = maxT / numTimeSteps;
    const float invTau = -1.0f / tau;

    // CPU vectors for debug dumps
    std::vector<float> dbgAccelRhsXy(velocitySolver->getRhs().size());
    std::vector<float> dbgVelocityXy(velocityXy.size());
    std::vector<float> dbgPressureRhs(pressureSolver.solver.getRhs().size());
    std::vector<float> dbgInternalP(ctx.internalPressureNodes.size());
    std::vector<float> dbgFullP(numPressureNodes);

    ProgressTracker progressTracker(numTimeSteps);

    cu::Event cuBegin;
    cu::Event cuConvection;
    cu::Event cuTentative;
    cu::Event cuPressure;
    cu::Event cuFinal;

    for (int iT = 0; iT <= numTimeSteps; iT++)
    {
        u::Stopwatch bigSw;
        u::Stopwatch sw;

        LogEntry currLog;
        currLog.id = iT;

        const float currTime = iT * tau;
        auto currOutput = outputHandler.getCurrentOutput(iT, currTime);

        const bool dumpNow = dbgDumps && (iT % dumpCfg.mod == 0);

        cuBegin.record();

        // Update convection
        auto & currConvection = fcSpmv.b;
        fcSpmv.compute(velocityXy, currConvection);

        cuConvection.record();

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

        // =========================================================================================
        // Find tentative velocity in three steps:
        // 1) Compute accelRhsXy = A * velocityXy
        // 2) Solve M0 * accelXy = accelRhsXy
        // 3) Apply the acceleration

        aSpmm.compute(velocityXy, accelRhs);
        if (dumpNow)
        {
            accelRhs.download(dbgAccelRhsXy);
            const std::string outFname = std::format("{}/{}_tentativeRhsXy.bin", dumpDir, iT);
            printDebugFileName(outFname);
            linalg::write(outFname, dbgAccelRhsXy);
        }

        dirichletAcceleration.sliceRhs(); // accelRhs -> accelRhsInt
        accelInt.memsetZero();
        velocitySolver->solve();
        dirichletAcceleration.scatterAcceleration(); // accelInt -> accel

        currLog.mseTentative[0] = velocitySolver->getLastMse(0);
        currLog.mseTentative[1] = velocitySolver->getLastMse(1);
        currLog.itersTentative = velocitySolver->getLastIterations();
        if (!std::isfinite(currLog.mseTentative[0]) || !std::isfinite(currLog.mseTentative[1]))
        {
            throw std::runtime_error("Tentative acceleration is bad");
        }
        // v* = v - tau * accel
        cu::saxpy(blas, 2 * numVelocityNodes, accel.get(), velocityXy.get(), -tau);

        cuTentative.record();
        if (dumpNow)
        {
            velocityXy.download(dbgVelocityXy);
            const std::string outFname = std::format("{}/{}_tentativeVxy.bin", dumpDir, iT);
            printDebugFileName(outFname);
            linalg::write(outFname, dbgVelocityXy);
        }
        currLog.tTentative = sw.millis(true);
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
        cuPressure.record();
        currLog.msePressure = pressureSolver.solver.getLastMse();
        currLog.itersPressure = pressureSolver.solver.getLastIterations();
        if (dumpNow)
        {
            // pressureSolver.rhs.download(dbgPressureRhs);
            pressureSolver.solver.getRhs().download(dbgPressureRhs);
            const std::string outFname = std::format("{}/{}_pressureRhs.bin", dumpDir, iT);
            printDebugFileName(outFname);
            linalg::write(outFname, dbgPressureRhs);
        }

        auto & pressure = pressureSolver.dense;
        assert(pressure.size() == numPressureNodes);

        if (dumpNow)
        {
            // pressureSolver.internalPressure.download(dbgInternalP);
            pressureSolver.solver.getSol().download(dbgInternalP);
            pressure.download(dbgFullP);

            std::string outFname = std::format("{}/{}_internalP.bin", dumpDir, iT);
            printDebugFileName(outFname);
            linalg::write(outFname, dbgInternalP);

            outFname = std::format("{}/{}_fullP.bin", dumpDir, iT);
            printDebugFileName(outFname);
            linalg::write(outFname, dbgFullP);
        }

        // Copy to output
        if (currOutput.pressure)
        {
            auto & outP = *currOutput.pressure;
            outP.resize(numPressureNodes);
            pressure.download(outP);
        }
        currLog.tPressure = sw.millis(true);
        // =========================================================================================

        // =========================================================================================
        // Find the final velocity by updating the tentative
        // (u_{i+1} - u_*) / tau = -nabla(p) <=>
        // <=> a = nabla(p) <=>
        // <=> (a, v) = (nabla(p), v)
        // Then update: u_{i+1} = u_* + tau * a
        // Calculate X and Y channels simultaneously

        // nablaPXy = pressureVelocityDiv * pressure;
        pvdSpmv.compute(pressure, nablaPXy);
        nablaPXy.copyTo(accelRhs); // TODO Can we compute in accelRhs directly?
        if (dumpNow)
        {
            accelRhs.download(dbgAccelRhsXy);
            const std::string outFname = std::format("{}/{}_finalRhsXy.bin", dumpDir, iT);
            printDebugFileName(outFname);
            linalg::write(outFname, dbgAccelRhsXy);
        }

        dirichletAcceleration.sliceRhs(); // accelRhs -> accelRhsInt
        accelInt.memsetZero();
        velocitySolver->solve();
        dirichletAcceleration.scatterAcceleration(); // accelInt -> accel

        currLog.mseFinal[0] = velocitySolver->getLastMse(0);
        currLog.mseFinal[1] = velocitySolver->getLastMse(1);
        currLog.itersFinal = velocitySolver->getLastIterations();
        if (!std::isfinite(currLog.mseFinal[0]) || !std::isfinite(currLog.mseFinal[1]))
        {
            throw std::runtime_error("Final acceleration is bad");
        }

        cu::saxpy(blas, 2 * numVelocityNodes, accel.get(), velocityXy.get(), -tau);

        cuFinal.record();

        if (dumpNow)
        {
            velocityXy.download(dbgVelocityXy);
            const std::string outFname = std::format("{}/{}_finalVxy.bin", dumpDir, iT);
            printDebugFileName(outFname);
            linalg::write(outFname, dbgVelocityXy);
        }

        // Copy to output
        if (currOutput.velocity)
        {
            auto & outVelocity = *currOutput.velocity;
            outVelocity.resize(velocityXy.size());
            velocityXy.download(outVelocity);
        }
        outputHandler.finishOutput(currOutput);

        currLog.tFinal = sw.millis();
        currLog.tTotal = bigSw.millis();

        cuFinal.sync();
        currLog.tCuConvection = cuConvection.elapsedTimeMs(cuBegin);
        currLog.tCuTentative = cuTentative.elapsedTimeMs(cuConvection);
        currLog.tCuPressure = cuPressure.elapsedTimeMs(cuTentative);
        currLog.tCuFinal = cuFinal.elapsedTimeMs(cuPressure);

        log.add(currLog);
        // =========================================================================================

        // Print progress info
        progressTracker.update(iT);
    }
}