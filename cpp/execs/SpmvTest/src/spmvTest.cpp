#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include <semaphore.h>
// #include <sched.h>

#include <linalg/csrMatrix.h>
#include <linalg/gaussSeidel.h>
#include <linalg/graphs.h>
#include <linalg/io.h>

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

enum class SpmvSignalType
{
    Start,
    Stop
};

struct SpmvSingal
{
    SpmvSignalType type;
};

template <typename F>
class ParallelSpmv
{
    const linalg::CsrMatrix<F> & m;

    std::vector<std::thread> threads;
    Semaphore semStart;
    Semaphore semDone;

    SpmvSingal signal;

    F * result;
    const F * v;

    void threadFunc(const int threadIdx)
    {
        const int nRows = m.rows;
        const int numThreads = threads.size();

        while (true)
        {
            semStart.wait();
            if (signal.type == SpmvSignalType::Stop)
            {
                semDone.post();
                break;
            }
            else if (signal.type == SpmvSignalType::Start)
            {
                for (int i = threadIdx; i < nRows; i += numThreads)
                {
                    const int j1 = m.rowStart[i + 1];
                    F sum = 0;
                    for (int j = m.rowStart[i]; j < j1; j++)
                    {
                        const int col = m.colIdx[j];
                        sum += m.values[j] * v[col];
                    }
                    result[i] = sum;
                    // result[i] = 0;
                }
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
    ParallelSpmv(const linalg::CsrMatrix<F> & matrix, const int nThreads)
        : m(matrix)
    {
        if (nThreads < 1 || nThreads > 64)
        {
            throw std::invalid_argument("Bad number of threads");
        }

        threads.resize(nThreads);
        for (int i = 0; i < nThreads; i++)
        {
            threads[i] = std::thread(&ParallelSpmv<F>::threadFunc, this, i);
        }
    }

    void rMult(const F * v, F * result)
    {
        this->v = v;
        this->result = result;

        signal.type = SpmvSignalType::Start;
        startAndSyncThreads();
    }

    template <linalg::VectorLike<F> A, linalg::VectorLike<F> B>
    void rMult(const A & src, B & dst)
    {
        if (src.size() != m.cols)
        {
            throw std::invalid_argument(std::format("{}: Bad size of src vector [{}] - expected {}",
                                                    __FUNCTION__, src.size(), m.cols));
        }
        if (dst.size() != m.rows)
        {
            throw std::invalid_argument(std::format("{}: Bad size of dst vector [{}] - expected {}",
                                                    __FUNCTION__, dst.size(), m.rows));
        }
        rMult(src.data(), dst.data());
    }

    ~ParallelSpmv()
    {
        signal.type = SpmvSignalType::Stop;
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
    const std::string usageMsg = "./SpmvTest <csr matrix>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    const std::string matrixFname = argv[1];

    auto m = linalg::readCsr<float>(matrixFname);

    const int nRows = m.rows;
    const int nCols = m.cols;

    std::vector<float> v(nCols), resultS(nRows), resultP(nRows);
    std::default_random_engine rng(1337151516);
    std::uniform_real_distribution<float> dist(-10, 10);
    for (int i = 0; i < nCols; i++)
    {
        v[i] = dist(rng);
    }

    const int nRuns = 20;
    std::cout << "Serial times:\n";
    double sum = 0;
    for (int i = 0; i < nRuns; i++)
    {
        u::Stopwatch sw;
        m.rMult(v, resultS);
        const auto currT = sw.millis();
        std::cout << currT << " ms\n";
        sum += currT;
    }
    const double avgSerial = sum / nRuns;

    const int threadCount = 4;
    ParallelSpmv<float> pSpmv(m, threadCount);

    std::cout << "Parallel times:\n";
    sum = 0;
    for (int i = 0; i < nRuns; i++)
    {
        u::Stopwatch sw;
        pSpmv.rMult(v, resultP);
        const auto currT = sw.millis();
        std::cout << currT << " ms\n";
        sum += currT;
    }
    const double avgParallel = sum / nRuns;

    std::cout << std::format("Average: serial = {} ms, parallel = {} ms\n", avgSerial, avgParallel);

    double sumSq = 0;
    float maxDelta = 0;
    for (int i = 0; i < nRows; i++)
    {
        const float delta = resultP[i] - resultS[i];
        if (std::abs(delta) > std::abs(maxDelta))
        {
            maxDelta = delta;
        }
        sumSq += delta * delta;
    }
    const float mse = std::sqrt(sumSq / nRows);
    std::cout << std::format("Max delta = {}, MSE = {}\n", maxDelta, mse);

    return 0;
}