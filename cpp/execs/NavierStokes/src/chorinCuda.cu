#include <NavierStokes/chorinCuda.h>

#include <cassert>
#include <format>
#include <stdexcept>

#include <cu/blas.h>
#include <cu/conjGradF.h>
#include <cu/csrF.h>
#include <cu/sparse.h>
#include <cu/spmv.h>

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

    float cgTarget = 1e-5;
    int cgMaxIters = 300;

    // Input/output buffer
    // Before pressure is calculated, this is tentativeVelDiv
    // After it is calculated, this is the pressure
    cu::vec<float> dense;
    cu::vec<int> internalIds;
    cu::vec<float> rhs;
    cu::vec<float> internalPressure;

    cusparseSpVecDescr_t sparseInput;  // values = rhs
    cusparseSpVecDescr_t sparseOutput; // values = internalPressure

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

        float mse = cg.solve(rhs, internalPressure, cgMaxIters, cgTarget);
        if (!std::isfinite(mse))
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
    auto blasRc = cublasSscal(blas.handle, origVelocityM1Vals.size(), &plusOne, origVelocityM1Vals.get(), 1);
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

    // Create descriptors for the X and Y velocities inside the big velocity vector
    // cusparseDnVecDescr_t velocityXDesc, velocityYDesc;
    // auto sparseRc = cusparseCreateDnVec(&velocityXDesc, numVelocityNodes,
    //                                     velocityXy.get(),
    //                                     cudaDataType::CUDA_R_32F);
    // if (sparseRc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
    // {
    //     throw std::runtime_error(std::format("cusparseCreateDnVec failed: {}", cusparseGetErrorName(sparseRc)));
    // }
    // sparseRc = cusparseCreateDnVec(&velocityYDesc, numVelocityNodes,
    //                                velocityXy.get() + numVelocityNodes,
    //                                cudaDataType::CUDA_R_32F);
    // if (sparseRc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
    // {
    //     throw std::runtime_error(std::format("cusparseCreateDnVec failed: {}", cusparseGetErrorName(sparseRc)));
    // }

    // Acceleration
    cu::vec<float> accel(numVelocityNodes);
    cu::ConjugateGradientF velocityMassCg(velocityMass);
    const float cgTargetAccel = 1e-5f;
    const int cgMaxItersAccel = 100;

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
    Solution result;
    result.steps.resize(numTimeSteps + 1);

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

        // =========================================================================================
        // Find tentative velocity in two steps:
        // 1) Compute accelRhsC = A * velocityC
        // 2) Solve M0 * accelC = accelRhsC
        // Repeat for C = {X, Y}

        // X
        auto & accelRhs = aSpmv.b;
        aSpmv.compute(velocityX, accelRhs);
        accel.memsetZero();
        float mse = velocityMassCg.solve(accelRhs, accel, cgMaxItersAccel, cgTargetAccel);
        if (!std::isfinite(mse))
        {
            throw std::runtime_error("Bad CG");
        }
        // v* = v - tau * accel
        cu::saxpy(blas, numVelocityNodes, accel.get(), velocityXy.get(), -tau);

        // Y
        aSpmv.compute(velocityY, accelRhs);
        accel.memsetZero();
        mse = velocityMassCg.solve(accelRhs, accel, cgMaxItersAccel, cgTargetAccel);
        if (!std::isfinite(mse))
        {
            throw std::runtime_error("Bad CG");
        }
        // v* = v - tau * accel
        cu::saxpy(blas, numVelocityNodes, accel.get(), velocityXy.get() + numVelocityNodes, -tau);

        // Reimpose BCs
        dirichletVelocity.impose();
        // =========================================================================================

        // =========================================================================================
        // Find the pressure:
        // 1) Compute the RHS of the system:
        //      tentativeVelDiv = velocityPressureDiv * tentativeVelocityXy;
        //      pressureRhs = gather(tenativeVelDiv, internalPressureNodes)
        // 2) Find the internal pressure: pressureInt = pressureStiffnessSolver.solve(pressureRhs);
        // 3) Scatter the internal pressure

        vpdSpmv.compute(velocityXy, pressureSolver.dense);
        pressureSolver.update();
        auto & pressure = pressureSolver.dense;
        assert(pressure.size() == numPressureNodes);

        // Copy to output
        auto & outP = result.steps[iT].pressure;
        outP.resize(numPressureNodes);
        pressure.download(outP);
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
        mse = velocityMassCg.solve(nablaPX, accel, cgMaxItersAccel, cgTargetAccel);
        if (!std::isfinite(mse))
        {
            throw std::runtime_error("Bad CG");
        }
        cu::saxpy(blas, numVelocityNodes, accel.get(), velocityX.get(), -tau);

        // Y
        accel.memsetZero();
        mse = velocityMassCg.solve(nablaPY, accel, cgMaxItersAccel, cgTargetAccel);
        if (!std::isfinite(mse))
        {
            throw std::runtime_error("Bad CG");
        }
        cu::saxpy(blas, numVelocityNodes, accel.get(), velocityY.get(), -tau);

        dirichletVelocity.impose();

        // Copy to output
        auto & outVelocity = result.steps[iT].velocity;
        outVelocity.resize(velocityXy.size());
        velocityXy.download(outVelocity);
        // =========================================================================================
    }

    return result;
}