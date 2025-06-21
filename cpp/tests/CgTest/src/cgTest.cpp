#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include <linalg/gaussSeidel.h>
#include <linalg/io.h>
#include <linalg/vectors.h>

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

template <typename V>
double conjugateGradientD(const linalg::CsrMatrix<float> & m, V & xIn, const V & bIn,
                          const int maxIters, const double target)
{
    const int n = m.cols;
    assert(n == m.rows);
    assert(n == x.size());
    assert(n == b.size());

    // Work vectors
    std::vector<double> x(n);
    std::vector<double> b(n);
    std::vector<double> r(n); // Residuals
    std::vector<double> p(n); // Direction
    std::vector<double> d(n); // M * p

    for (int i = 0; i < n; i++)
    {
        x[i] = xIn[i];
        b[i] = bIn[i];
    }

    // Init: r = b - Mx
    auto initR = [&]()
    {
        m.rMultD(x, r);
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

        m.rMultD(p, d);

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

    for (int i = 0; i < n; i++)
    {
        xIn[i] = x[i];
    }

    return lastMse;
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

    std::vector<float> xCg(n, 0);
    const double target = 1e-6;
    const double cgRes = conjugateGradient(m, xCg, rhs, std::min<int>(n, 200), target);
    const double cgMse = m.mse(xCg, rhs);
    std::cout << "cgRes = " << cgRes << ", cgMse = " << cgMse << "\n";

    std::vector<float> x(n, 0);
    linalg::gaussSeidel(m, x, rhs, 300, target);

    const double newMse = m.mse(x, rhs);
    std::cout << "Solved MSE (Gauss Seidel): " << newMse << "\n";

    return 0;
}