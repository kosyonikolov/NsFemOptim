#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <string>
#include <thread>

#include <semaphore.h>
// #include <sched.h>

#include <linalg/csrMatrix.h>
#include <linalg/graphs.h>
#include <linalg/io.h>
#include <linalg/gaussSeidel.h>

#include <utils/stopwatch.h>

class Semaphore
{
    sem_t sem;

public:
    Semaphore(const int value = 0)
    {
        auto rc = sem_init(&sem, 0, value);
        if (rc != 0)
        {
            const int err = errno;
            throw std::runtime_error("Failed to init semaphore: " + std::string(strerror(err)));
        }
    }

    Semaphore(const Semaphore & other) = delete;
    Semaphore & operator=(const Semaphore & other) = delete;

    void wait()
    {
        [[maybe_unused]] auto rc = sem_wait(&sem);
        assert(rc == 0);
    }

    void post()
    {
        [[maybe_unused]] auto rc = sem_post(&sem);
        assert(rc == 0);
    }

    ~Semaphore()
    {
        auto rc = sem_destroy(&sem);
        if (rc != 0)
        {
            const int err = errno;
            std::cerr << "Failed to destroy semaphore: " << strerror(err) << "\n";
        }
    }
};

enum class GspSignalType
{
    WorkOnPartition,
    CalculateMse,
    Stop
};

struct GspSignal
{
    GspSignalType type;
    int partitionId;
};

template <typename F>
class ParallelGaussSeidel2ch
{
    const linalg::CsrMatrix<F> & m;
    const std::vector<std::vector<int>> partitions;

    std::vector<std::thread> threads;
    Semaphore semStart;
    Semaphore semDone;

    std::vector<double> threadSqSums0; // X channel
    std::vector<double> threadSqSums1; // Y channel
    GspSignal signal;

    F * x;
    const F * b;

    void threadFunc(const int threadIdx)
    {
        // Not significant improvement in median times
        // Some improvement for max times
        // cpu_set_t cpuSet;
        // CPU_ZERO(&cpuSet);
        // CPU_SET(2 * threadIdx, &cpuSet);
        // if (sched_setaffinity(getpid(), sizeof(cpuSet), &cpuSet) == -1)
        // {
        //     const int err = errno;
        //     std::cerr << "setaffinity failed: " << strerror(err) << "\n";
        // }

        const int nRows = m.rows;
        const int numThreads = threads.size();

        while (true)
        {
            semStart.wait();
            if (signal.type == GspSignalType::Stop)
            {
                semDone.post();
                break;
            }
            else if (signal.type == GspSignalType::WorkOnPartition)
            {
                const int id = signal.partitionId;
                assert(id >= 0 && id < partitions.size());
                const auto & p = partitions[id];
                const int limit = p.size();

                for (int k = threadIdx; k < limit; k += numThreads)
                {
                    const int i = p[k]; // Row

                    const int j1 = m.rowStart[i + 1];
                    F diag = 0; // element at (i, i)
                    double negSum0 = 0;
                    double negSum1 = 0;
                    for (int j = m.rowStart[i]; j < j1; j++)
                    {
                        const int col = m.column[j];
                        if (col == i)
                        {
                            diag = m.values[j];
                        }
                        else
                        {
                            negSum0 += m.values[j] * x[2 * col + 0];
                            negSum1 += m.values[j] * x[2 * col + 1];
                        }
                    }

                    assert(diag != 0);
                    x[2 * i + 0] = (b[2 * i + 0] - negSum0) / diag;
                    x[2 * i + 1] = (b[2 * i + 1] - negSum1) / diag;
                }
            }
            else if (signal.type == GspSignalType::CalculateMse)
            {
                std::array<double, 2> errSum = {0, 0};
                for (int i = threadIdx; i < nRows; i += numThreads)
                {
                    const int j1 = m.rowStart[i + 1];
                    std::array<F, 2> sum = {0, 0};
                    for (int j = m.rowStart[i]; j < j1; j++)
                    {
                        const int col = m.column[j];
                        sum[0] += m.values[j] * x[2 * col + 0];
                        sum[1] += m.values[j] * x[2 * col + 1];
                    }
                    const auto delta0 = sum[0] - b[2 * i + 0];
                    const auto delta1 = sum[1] - b[2 * i + 1];
                    errSum[0] += delta0 * delta0;
                    errSum[1] += delta1 * delta1;
                }

                threadSqSums0[threadIdx] = errSum[0];
                threadSqSums1[threadIdx] = errSum[1];
            }
            else
            {
                std::cerr << "!!! Unknown signal received: " << static_cast<int>(signal.type) << " !!!\n";
            }

            semDone.post();
        }
    }

    void startAndSyncThreads()
    {
        const int n = threads.size();
        for (int i = 0; i < n; i++)
        {
            semStart.post();
        }

        for (int i = 0; i < n; i++)
        {
            semDone.wait();
        }
    }

public:
    ParallelGaussSeidel2ch(const linalg::CsrMatrix<F> & matrix, const std::vector<std::vector<int>> & parts,
                           const int nThreads)
        : m(matrix), partitions(parts)
    {
        if (nThreads < 1 || nThreads > 64)
        {
            throw std::invalid_argument("Bad number of threads");
        }

        threadSqSums0.resize(nThreads);
        threadSqSums1.resize(nThreads);
        threads.resize(nThreads);
        for (int i = 0; i < nThreads; i++)
        {
            threads[i] = std::thread(&ParallelGaussSeidel2ch<F>::threadFunc, this, i);
        }
    }

    std::tuple<double, double> solve(F * x, const F * b, const int maxIters, const double eps)
    {
        this->x = x;
        this->b = b;

        const int numThreads = threads.size();
        const int nP = partitions.size();
        const int nRows = m.rows;

        double lastMse0 = -1;
        double lastMse1 = -1;

        for (int i = 0; i < maxIters; i++)
        {
            // Process the elements of each partition in parallel
            signal.type = GspSignalType::WorkOnPartition;
            for (int p = 0; p < nP; p++)
            {
                signal.partitionId = p;
                startAndSyncThreads();
            }

            // Calculate MSE
            signal.type = GspSignalType::CalculateMse;
            startAndSyncThreads();

            double sumSq0 = 0;
            double sumSq1 = 0;
            for (int t = 0; t < numThreads; t++)
            {
                sumSq0 += threadSqSums0[t];
                sumSq1 += threadSqSums1[t];
            }
            lastMse0 = std::sqrt(sumSq0 / nRows);
            lastMse1 = std::sqrt(sumSq1 / nRows);
            // std::cout << std::format("{}: {}, {}\n", i, lastMse0, lastMse1);
            if (lastMse0 < eps && lastMse1 < eps)
            {
                break;
            }
        }

        return {lastMse0, lastMse1};
    }

    ~ParallelGaussSeidel2ch()
    {
        signal.type = GspSignalType::Stop;
        startAndSyncThreads();
        const int numThreads = threads.size();
        for (int i = 0; i < numThreads; i++)
        {
            threads[i].join();
        }
    }
};

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./GsTest <csr matrix> <x0> <b>";
    if (argc != 4)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string matrixFname = argv[1];
    const std::string xFname = argv[2];
    const std::string bFname = argv[3];

    auto m = linalg::readCsr<float>(matrixFname);
    auto x0 = linalg::readVec<float>(xFname);
    auto b = linalg::readVec<float>(bFname);

    if (m.cols != m.rows)
    {
        std::cerr << "Matrix is not square\n";
    }

    if (2 * m.rows != x0.size() || 2 * m.rows != b.size())
    {
        std::cerr << "Incompatible sizes\n";
    }

    auto graph = linalg::buildCsrGraph(m);
    auto partition = linalg::partitionGraphDSatur(graph);

    std::cout << partition.size() << " partitions:\n";
    for (int i = 0; i < partition.size(); i++)
    {
        std::cout << i << ": " << partition[i].size() << "\n";
    }

    auto xP = x0;
    auto xS = x0;

    const int maxIters = 100;
    const double eps = 1e-9;
    const int nRuns = 100;
    std::cout << "Serial times:\n";
    for (int i = 0; i < nRuns; i++)
    {
        xS = x0;
        u::Stopwatch sw;
        linalg::gaussSeidel2ch(m, xS, b, maxIters, eps);
        std::cout << sw.millis() << " ms\n";
    }

    const int nThreads = 8;
    ParallelGaussSeidel2ch<float> solver(m, partition, nThreads);

    std::cout << "Parallel times:\n";
    for (int i = 0; i < nRuns; i++)
    {
        xP = x0;
        u::Stopwatch sw;
        solver.solve(xP.data(), b.data(), maxIters, eps);
        std::cout << sw.millis() << " ms\n";
    }

    float maxDelta = 0;
    double sqSum = 0;
    const int n = xS.size();
    for (int i = 0; i < n; i++)
    {
        const float delta = xP[i] - xS[i];
        if (std::abs(delta) > std::abs(maxDelta))
        {
            maxDelta = delta;
        }
        sqSum += delta * delta;
    }

    const double mse = std::sqrt(sqSum / n);
    std::cout << "MSE: " << mse << "\n";
    std::cout << "Max delta: " << maxDelta << "\n";

    double mseS0, mseS1;
    double mseP0, mseP1;
    linalg::mse2ch(m, xS.data(), b.data(), mseS0, mseS1);
    linalg::mse2ch(m, xP.data(), b.data(), mseP0, mseP1);

    std::cout << "SysMSE serial: " << mseS0 << ", " << mseS1 << "\n";
    std::cout << "SysMSE parallel: " << mseP0 << ", " << mseP1 << "\n";

    return 0;
}