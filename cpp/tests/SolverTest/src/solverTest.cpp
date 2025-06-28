#include <cassert>
#include <iostream>
#include <string>
#include <vector>

#include <linalg/gaussSeidel.h>
#include <linalg/graphs.h>
#include <linalg/io.h>
#include <linalg/jacobi.h>
#include <linalg/vectors.h>

#include <cu/gaussSeidelHost.h>
#include <cu/conjGradHost.h>

#include <utils/stopwatch.h>

using AlgoFn = std::vector<float> (*)(const linalg::CsrMatrix<float> &, const std::vector<float> &, const int, const float);

template <typename F>
double conjugateGradient(const linalg::CsrMatrix<F> & m, std::vector<F> & x, const std::vector<F> & b,
                         const int maxIters, const double target)
{
    const int n = m.cols;
    assert(n == m.rows);
    assert(n == x.size());
    assert(n == b.size());

    // Work vectors
    std::vector<F> r(n); // Residuals
    std::vector<F> p(n); // Direction
    std::vector<F> d(n); // M * p

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
        std::cout << iter << ": " << currMse << "\n";
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

std::vector<float> jacobi(const linalg::CsrMatrix<float> & m, const std::vector<float> & rhs,
                          const int maxIters, const float eps)
{
    std::vector<float> x(rhs.size(), 0);
    std::vector<float> aux(rhs.size());
    linalg::jacobi(m, x, rhs, maxIters, eps, aux);
    return x;
}

std::vector<float> gaussSeidel(const linalg::CsrMatrix<float> & m, const std::vector<float> & rhs,
                               const int maxIters, const float eps)
{
    std::vector<float> x(rhs.size(), 0);
    linalg::gaussSeidel(m, x, rhs, maxIters, eps);
    return x;
}

// Uses the coloring order of the CUDA version
std::vector<float> gaussSeidelCustom(const linalg::CsrMatrix<float> & m, const std::vector<float> & rhs,
                                     const int maxIters, const float eps)
{
    assert(m.cols == m.rows);
    const int n = m.cols;

    // Create a coloring of the matrix
    // Use the smallest-last ordering for now - it seems to produce good results
    auto graph = linalg::buildCsrGraph(m);
    assert(graph.size() == n);
    auto slOrder = linalg::buildSmallestLastOrdering(graph);
    auto parts = linalg::partitionGraphGreedy(graph, slOrder);

    const int nColors = parts.size();

    // Sort the individual partitions and place them in the coloring vector
    std::vector<int> cpuColoring(n);
    int i = 0;
    for (int c = 0; c < nColors; c++)
    {
        auto & p = parts[c];
        std::sort(p.begin(), p.end());
        std::copy_n(p.begin(), p.size(), cpuColoring.begin() + i);
        i += p.size();
    }

    std::vector<float> x(rhs.size(), 0);
    linalg::gaussSeidelCustomOrder(m, x, rhs, cpuColoring, maxIters, eps);
    return x;
}

std::vector<float> gaussSeidelCuda(const linalg::CsrMatrix<float> & m, const std::vector<float> & rhs,
                                   const int maxIters, const float eps)
{
    cu::GaussSeidelHost gs(m);

    std::vector<float> x(rhs.size(), 0);
    gs.solve(rhs, x, maxIters, eps);

    return x;
}

std::vector<float> cg(const linalg::CsrMatrix<float> & m, const std::vector<float> & rhs,
                      const int maxIters, const float eps)
{
    std::vector<float> x(rhs.size(), 0);
    conjugateGradient(m, x, rhs, maxIters, eps);
    return x;
}

std::vector<float> cgCuda(const linalg::CsrMatrix<float> & m, const std::vector<float> & rhs,
                          const int maxIters, const float eps)
{
    cu::ConjGradHost cg(m);

    std::vector<float> x(rhs.size(), 0);
    cg.solve(rhs, x, maxIters, eps);

    return x;
}

template <typename Dst, typename Src>
std::vector<Dst> convert(const std::vector<Src> & v)
{
    std::vector<Dst> result(v.size());
    for (size_t i = 0; i < v.size(); i++)
    {
        result[i] = v[i];
    }
    return result;
}

std::vector<float> cgd(const linalg::CsrMatrix<float> & m, const std::vector<float> & rhs,
                       const int maxIters, const float eps)
{
    linalg::CsrMatrix<double> mD;
    mD.rows = m.rows;
    mD.cols = m.cols;
    mD.rowStart = m.rowStart;
    mD.column = m.column;
    mD.values = convert<double>(m.values);

    std::vector<double> rhsD = convert<double>(rhs);
    std::vector<double> xD(rhs.size(), 0);
    conjugateGradient(mD, xD, rhsD, maxIters, eps);

    std::vector<float> x = convert<float>(xD);
    return x;
}

std::vector<std::vector<float>> splitChannels(const std::vector<float> & src, const int numCh)
{
    const int n = src.size() / numCh;
    assert(n * numCh == src.size());
    std::vector<std::vector<float>> result(numCh);
    for (int c = 0; c < numCh; c++)
    {
        result[c].resize(n);
    }

    for (int i = 0; i < n; i++)
    {
        for (int c = 0; c < numCh; c++)
        {
            result[c][i] = src[i * numCh + c];
        }
    }

    return result;
}

AlgoFn selectAlgo(const std::string & name)
{
#define RETIF(x)    \
    if (name == #x) \
        return x;
    RETIF(cg);
    RETIF(cgCuda);
    RETIF(cgd);
    RETIF(gaussSeidel);
    RETIF(gaussSeidelCuda);
    RETIF(gaussSeidelCustom);
    RETIF(jacobi);
#undef RETIF
    return 0;
}

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./SolverTest <algo> <matrix> <rhs>";
    if (argc != 4)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string algoName = argv[1];
    const std::string matFname = argv[2];
    const std::string rhsFname = argv[3];

    const auto m = linalg::readCsr<float>(matFname);
    const auto rhs = linalg::readVec<float>(rhsFname);

    std::cout << "Matrix size = " << m.rows << "x" << m.cols << "\n";
    std::cout << "RHS size = " << rhs.size() << "\n";

    if (m.rows != m.cols)
    {
        std::cerr << "Only square matrices supported for now\n";
        return 1;
    }
    const int n = m.rows;
    const int numCh = rhs.size() / n;
    if (numCh * n != rhs.size())
    {
        std::cerr << "Failed to determine number of channels\n";
        return 1;
    }
    std::cout << "Number of channels = " << numCh << "\n";

    auto channels = splitChannels(rhs, numCh);

    AlgoFn algo = selectAlgo(algoName);
    if (!algo)
    {
        std::cerr << "No algorithm matching " << algoName << "\n";
        return 1;
    }

    const int maxIters = 200;
    const float eps = 1e-9;

    std::vector<float> x;
    for (int c = 0; c < numCh; c++)
    {
        std::cout << "================ Channel " << c << " ================\n";
        auto & currRhs = channels[c];
        x = algo(m, currRhs, maxIters, eps);
        const double mse = m.mse(x, currRhs);
        std::cout << "Final MSE = " << mse << "\n";
    }

    return 0;
}