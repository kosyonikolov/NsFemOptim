#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include <cudss.h>

#include <Eigen/Sparse>

#include <linalg/eigen.h>
#include <linalg/gaussSeidel.h>
#include <linalg/io.h>
#include <linalg/vectors.h>

#include <cu/conjGradF.h>
#include <cu/vec.h>

#include <utils/stopwatch.h>

template <typename F, u::VectorLike<F> V>
double conjugateGradient(const linalg::CsrMatrix<F> & m, V & x, const V & b,
                         const int maxIters, const double target)
{
    const int n = m.cols;
    assert(n == m.rows);
    assert(n == x.size());
    assert(n == b.size());

    // Work vectors
    std::vector<float> r(n); // Residuals
    std::vector<float> p(n); // Direction
    std::vector<float> d(n); // M * p

    // Init: r = b - Mx
    auto initR = [&]()
    {
        m.rMult(x, r);
        double sum = 0;
        for (int i = 0; i < n; i++)
        {
            r[i] = b[i] - r[i];
            sum += r[i] * r[i];
        }
        return sum;
    };

    double dotR0 = initR();
    p = r;
    double lastMse = std::numeric_limits<double>::infinity();

    for (int iter = 0; iter < maxIters; iter++)
    {
        // if (iter > 0 && iter % 100 == 0)
        // {
        //     dotR0 = initR();
        //     p = r;
        // }

        m.rMult(p, d);

        const double dotDp = linalg::dot(d, p);
        const double alpha = dotR0 / dotDp;

        // Update x and r
        for (int i = 0; i < n; i++)
        {
            x[i] += alpha * p[i];
            r[i] -= alpha * d[i];
        }

        const double dotR1 = linalg::dot(r, r);
        const double currMse = std::sqrt(dotR1 / n);
        std::cout << "[CG] " << iter << ": " << currMse << "\n";
        lastMse = currMse;

        // Check for convergence
        if (currMse <= target)
        {
            break;
        }

        // Update direction
        const double beta = dotR1 / dotR0;
        for (int i = 0; i < n; i++)
        {
            p[i] = r[i] + beta * p[i];
        }
        dotR0 = dotR1;
    }

    return lastMse;
}

float cgCudaHost(const linalg::CsrMatrix<float> & m, std::vector<float> & x, const std::vector<float> & b,
                 const int maxIters, const float target)
{
    cu::csrF gpuMat(m);
    cu::vec<float> gpuX(x);
    cu::vec<float> gpuB(b);

    cu::ConjugateGradientF cg(gpuMat);
    auto res = cg.solve(gpuB, gpuX, maxIters, target);
    gpuX.download(x);
    return res;
}

float solveCudssHost(const linalg::CsrMatrix<float> & m, std::vector<float> & x, const std::vector<float> & b,
                     const int maxIters, const float target)
{
    if (m.rows != m.cols)
    {
        throw std::invalid_argument("Only square matrices supported");
    }
    const int n = m.rows;
    if (x.size() != n)
    {
        throw std::invalid_argument("Bad X size");
    }
    if (b.size() != n)
    {
        throw std::invalid_argument("Bad B size");
    }

    cu::csrF gpuMat(m);
    cu::vec<float> gpuX(x);
    cu::vec<float> gpuB(b);

    cudssHandle_t handle;

    auto rc = cudssCreate(&handle);
    if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::format("cudssCreate failed: {}", static_cast<int>(rc)));
    }

    // Stream goes here

    /* Creating cuDSS solver configuration and data objects */
    cudssConfig_t solverConfig;
    cudssData_t solverData;

    rc = cudssConfigCreate(&solverConfig);
    if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::format("cudssConfigCreate failed: {}", static_cast<int>(rc)));
    }

    rc = cudssDataCreate(handle, &solverData);
    if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::format("cudssDataCreate failed: {}", static_cast<int>(rc)));
    }

    /* Create matrix objects for the right-hand side b and solution x (as dense matrices). */
    cudssMatrix_t xMat, bMat;

    rc = cudssMatrixCreateDn(&bMat, n, 1, n, gpuB.get(), cudaDataType::CUDA_R_32F, cudssLayout_t::CUDSS_LAYOUT_COL_MAJOR);
    if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::format("cudssMatrixCreateDn failed for b: {}", static_cast<int>(rc)));
    }
    rc = cudssMatrixCreateDn(&xMat, n, 1, n, gpuX.get(), cudaDataType::CUDA_R_32F, cudssLayout_t::CUDSS_LAYOUT_COL_MAJOR);
    if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::format("cudssMatrixCreateDn failed for x: {}", static_cast<int>(rc)));
    }

    /* Create a matrix object for the sparse input matrix. */
    cudssMatrix_t A;
    rc = cudssMatrixCreateCsr(&A, n, n, gpuMat.values.size(),
                              gpuMat.rowStart.get(), 0,
                              gpuMat.column.get(),
                              gpuMat.values.get(),
                              cudaDataType_t::CUDA_R_32I,
                              cudaDataType_t::CUDA_R_32F,
                              cudssMatrixType_t::CUDSS_MTYPE_SPD,
                              cudssMatrixViewType_t::CUDSS_MVIEW_FULL,
                              cudssIndexBase_t::CUDSS_BASE_ZERO);
    if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::format("cudssMatrixCreateCsr failed: {}", static_cast<int>(rc)));
    }

    u::Stopwatch sw;

    rc = cudssExecute(handle, CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, xMat, bMat);
    if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::format("CUDSS_PHASE_ANALYSIS failed: {}", static_cast<int>(rc)));
    }

    const auto tAnalysis = sw.millis(true);

    /* Factorization */
    rc = cudssExecute(handle, CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, xMat, bMat);
    if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
    {
        throw std::runtime_error(std::format("CUDSS_PHASE_FACTORIZATION failed: {}", static_cast<int>(rc)));
    }

    const auto tFactorization = sw.millis(true);

    std::cout << std::format("CuDSS times: analysis = {}, factorization = {}\n",
                             tAnalysis, tFactorization);

    const int numRuns = 20000;
    for (int i = 0; i < numRuns; i++)
    {
        sw.reset();
        /* Solving */
        rc = cudssExecute(handle, CUDSS_PHASE_SOLVE, solverConfig, solverData, A, xMat, bMat);
        if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("CUDSS_PHASE_SOLVE failed: {}", static_cast<int>(rc)));
        }
        gpuX.download(x);
        const auto tSolve = sw.millis(true);
        std::cout << "Solve = " << tSolve << " ms\n";
    }

    return 0;
}

float solveEigen(const linalg::CsrMatrix<float> & m, std::vector<float> & x, const std::vector<float> & b,
                 const int maxIters, const float target)
{
    using SpMat = Eigen::SparseMatrix<float, Eigen::RowMajor>;
    using Vector = Eigen::Vector<float, Eigen::Dynamic>;

    auto eigenMat = linalg::eigenFromCsr(m);
    Eigen::SimplicialLDLT<SpMat> solver(eigenMat);
    Vector eigenB(b.size());
    for (int i = 0; i < b.size(); i++)
    {
        eigenB[i] = b[i];
    }
    Vector eigenX(x.size());
    eigenX.setZero();

    const int numRuns = 100;
    for (int i = 0; i < numRuns; i++)
    {
        u::Stopwatch sw;
        eigenX = solver.solve(eigenB);
        const auto t = sw.millis();
        std::cout << "Solve = " << t << " ms\n";
    }

    for (int i = 0; i < x.size(); i++)
    {
        x[i] = eigenX[i];
    }

    return 0;
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./CgTest <matrix> <rhs> <sol>";
    if (argc != 4)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string matFname = argv[1];
    const std::string rhsFname = argv[2];
    const std::string solFname = argv[3];

    const auto m = linalg::readCsr<float>(matFname);
    const auto rhs = linalg::readVec<float>(rhsFname);
    const auto sol = linalg::readVec<float>(solFname);

    std::cout << std::format("Matrix = {}x{}, rhs = {}, sol = {}\n", m.rows, m.cols, rhs.size(), sol.size());
    if (m.cols != m.rows || m.cols != rhs.size() || m.cols != sol.size())
    {
        std::cerr << "Mismatching dimensions\n";
    }

    const double rhsNorm = linalg::normL2<float>(rhs, true);
    const double solNorm = linalg::normL2<float>(sol, false);
    const double groundTruthMse = m.mse(sol, rhs);
    std::cout << "Rhs norm: " << rhsNorm << "\n";
    std::cout << "Sol norm: " << solNorm << "\n";
    std::cout << "Ground-truth MSE: " << groundTruthMse << "\n";

    const auto n = sol.size();

    const double target = 1e-6;
    const int maxIters = std::min<int>(n, 200);

    std::vector<float> xCgGpu(n, 0);
    const double cgResGpu = solveCudssHost(m, xCgGpu, rhs, maxIters, target);
    const double cgMseGpu = m.mse(xCgGpu, rhs);
    std::cout << "cgResGpu = " << cgResGpu << ", cgMseGpu = " << cgMseGpu << "\n";

    return 0;

    std::vector<float> xCg(n, 0);
    const double cgRes = conjugateGradient(m, xCg, rhs, maxIters, target);
    const double cgMse = m.mse(xCg, rhs);
    std::cout << "cgRes = " << cgRes << ", cgMse = " << cgMse << "\n";

    std::vector<float> x(n, 0);
    linalg::gaussSeidel(m, x, rhs, 300, target);

    const double newMse = m.mse(x, rhs);
    std::cout << "Solved MSE (Gauss Seidel): " << newMse << "\n";

    return 0;
}