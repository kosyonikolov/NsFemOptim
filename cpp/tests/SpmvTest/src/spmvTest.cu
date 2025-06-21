#include <cassert>
#include <cmath>
#include <cstring>
#include <iostream>
#include <random>
#include <string>
#include <thread>

#include <cusparse.h>

#include <semaphore.h>
// #include <sched.h>

#include <linalg/csrMatrix.h>
#include <linalg/gaussSeidel.h>
#include <linalg/graphs.h>
#include <linalg/io.h>
#include <linalg/vectors.h>

#include <utils/stopwatch.h>

#include <cu/blas.h>
#include <cu/csrF.h>
#include <cu/sparse.h>
#include <cu/spmv.h>
#include <cu/vec.h>

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
                        const int col = m.column[j];
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

    template <u::VectorLike<F> A, u::VectorLike<F> B>
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

void testMemorySpeed()
{
    const int n = 1 << 30;
    std::vector<int> src(n);
    std::vector<int> dst(n);
    for (int i = 0; i < n; i++)
    {
        src[i] = i * i;
    }

    const auto timeToSpeedGbs = [&](const float ms)
    {
        const float s = ms / 1000.0f;
        return n * sizeof(int) / (s * 1e9);
    };

    const int nRuns = 10;
    double sum = 0;
    for (int i = 0; i < nRuns; i++)
    {
        u::Stopwatch sw;
        std::copy_n(src.data(), n, dst.data());
        const auto t = sw.millis();
        sum += t;
        std::cout << t << " ms (" << timeToSpeedGbs(t) << " GB/s)\n";
    }
    const float avg = sum / nRuns;
    std::cout << "Average: " << avg << " ms (" << timeToSpeedGbs(avg) << " GB/s)\n";

    int test = 0;
    for (int i = 0; i < n; i++)
    {
        test += dst[i];
    }
    std::cout << "Test val = " << test << "\n";
}

void testCuCsrFSpmv(const linalg::CsrMatrix<float> & m)
{
    const int nRows = m.rows;
    const int nCols = m.cols;

    std::vector<float> src(nCols);
    std::vector<float> dstCpu(nRows), dstGpu(nRows);

    std::default_random_engine rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-50.0f, 10.0f);

    cu::csrF csrF(m);
    cu::Sparse lib;
    cu::spmv spmv(lib.handle(), csrF);

    const int nRuns = 10;
    for (int i = 0; i < nRuns; i++)
    {
        for (int j = 0; j < src.size(); j++)
        {
            src[j] = dist(rng);
        }

        // Cpu calc
        u::Stopwatch sw;
        m.rMult(src, dstCpu);
        const auto tCpu = sw.millis(true);

        // Gpu calc
        spmv.x.upload(src);
        const auto tUp = sw.millis(true);
        spmv.compute();
        const auto tCuda = sw.millis(true);
        spmv.b.download(dstGpu);
        const auto tDown = sw.millis();

        float maxDelta = 0;
        double sumSq = 0;
        for (int j = 0; j < dstCpu.size(); j++)
        {
            const float delta = dstGpu[j] - dstCpu[j];
            if (std::abs(delta) > std::abs(maxDelta))
            {
                maxDelta = delta;
            }
            sumSq += delta * delta;
        }

        const float avgErr = std::sqrt(sumSq / dstCpu.size());
        std::cout << std::format("{} / {}: CPU = {} ms; up = {} ms, CUDA = {} ms, down = {} ms\n",
                                 i + 1, nRuns, tCpu, tUp, tCuda, tDown);
        std::cout << std::format("\tAvg err = {}, max delta = {}\n", avgErr, maxDelta);
    }
}

void testDot(const int n)
{
    std::default_random_engine rng(std::random_device{}());
    std::uniform_real_distribution<float> dist(-1, 1);

    std::vector<float> cpuVec(n);
    cu::vec<float> gpuVec(n);

    cu::Blas blas;

    const int numRuns = 10;
    for (int k = 0; k < numRuns; k++)
    {
        for (int i = 0; i < n; i++)
        {
            cpuVec[i] = dist(rng);
        }
        gpuVec.upload(cpuVec);

        u::Stopwatch sw;
        const float dotCpu = linalg::dot(cpuVec, cpuVec);
        const auto tCpu = sw.millis();

        sw.reset();
        float dotGpu = 0;
        auto rc = cublasSdot(blas.handle, n, gpuVec.get(), 1, gpuVec.get(), 1, &dotGpu);
        const auto tGpu = sw.millis();
        if (rc != CUBLAS_STATUS_SUCCESS)
        {
            std::cerr << "!!! Cublas error !!!\n";
        }

        std::cout << std::format("{} / {}: cpu = {} ({} ms), gpu = {} ({} ms)\n", k + 1, numRuns,
                                 dotCpu, tCpu, dotGpu, tGpu);
    }
}

int main(int argc, char ** argv)
{
    testDot(1 << 20);
    return 0;

    const std::string usageMsg = "./SpmvTest <csr matrix>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    // testMemorySpeed();

    const std::string matrixFname = argv[1];

    auto m = linalg::readCsr<float>(matrixFname);

    testCuCsrFSpmv(m);

    return 0;
}