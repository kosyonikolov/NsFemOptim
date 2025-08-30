#include <NavierStokes/chorinCuda.h>

#include <cassert>
#include <filesystem>
#include <format>
#include <stdexcept>

#include <cu/blas.h>
#include <cu/csrF.h>
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
        DUMP(velocityMass);
        DUMP(velocityStiffness);
        DUMP(pressureStiffness);
        DUMP(pressureStiffnessInternal);
        DUMP(velocityPressureDiv);
        DUMP(pressureVelocityDiv);
        DUMP(fastConvectionIntegration);
#undef DUMP
    }

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
    // cu::vec<float> accel(2 * numVelocityNodes);
    const auto & vSolverCfg = cfg.velocitySolver;
    auto velocitySolver = cu::createSolver(vSolverCfg.method, 2,
                                           ctx.velocityMass, vSolverCfg.maxIterations,
                                           vSolverCfg.targetMse, vSolverCfg.mseCheckInterval);
    auto & accel = velocitySolver->getSol();

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

    for (int iT = 0; iT <= numTimeSteps; iT++)
    {
        u::Stopwatch bigSw;
        u::Stopwatch sw;

        LogEntry currLog;
        currLog.id = iT;

        const float currTime = iT * tau;
        auto currOutput = outputHandler.getCurrentOutput(iT, currTime);

        const bool dumpNow = dbgDumps && (iT % dumpCfg.mod == 0);

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

        // =========================================================================================
        // Find tentative velocity in three steps:
        // 1) Compute accelRhsXy = A * velocityXy
        // 2) Solve M0 * accelXy = accelRhsXy
        // 3) Apply the acceleration

        auto & accelRhs = velocitySolver->getRhs();
        aSpmm.compute(velocityXy, accelRhs);
        if (dumpNow)
        {
            accelRhs.download(dbgAccelRhsXy);
            const std::string outFname = std::format("{}/{}_tentativeRhsXy.bin", dumpDir, iT);
            printDebugFileName(outFname);
            linalg::write(outFname, dbgAccelRhsXy);
        }

        accel.memsetZero();
        velocitySolver->solve();
        currLog.mseTentative[0] = velocitySolver->getLastMse(0);
        currLog.mseTentative[1] = velocitySolver->getLastMse(1);
        currLog.itersTentative = velocitySolver->getLastIterations();
        if (!std::isfinite(currLog.mseTentative[0]) || !std::isfinite(currLog.mseTentative[1]))
        {
            throw std::runtime_error("Tentative acceleration is bad");
        }
        // v* = v - tau * accel
        cu::saxpy(blas, 2 * numVelocityNodes, accel.get(), velocityXy.get(), -tau);

        // Reimpose BCs
        dirichletVelocity.impose();
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

        accel.memsetZero();
        velocitySolver->solve();
        currLog.mseFinal[0] = velocitySolver->getLastMse(0);
        currLog.mseFinal[1] = velocitySolver->getLastMse(1);
        currLog.itersFinal = velocitySolver->getLastIterations();
        if (!std::isfinite(currLog.mseFinal[0]) || !std::isfinite(currLog.mseFinal[1]))
        {
            throw std::runtime_error("Final acceleration is bad");
        }

        cu::saxpy(blas, 2 * numVelocityNodes, accel.get(), velocityXy.get(), -tau);

        dirichletVelocity.impose();

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
        log.add(currLog);
        // =========================================================================================

        // Print progress info
        progressTracker.update(iT);
    }
}