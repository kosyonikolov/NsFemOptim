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

#include <utils/stopwatch.h>

#include <cu/csr.h>
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

struct CuSpmv
{
    const static auto op = cusparseOperation_t::CUSPARSE_OPERATION_NON_TRANSPOSE;

    cusparseHandle_t handle;

    cu::vec<float> src, dst;
    cu::csr<float> mat;
    cu::vec<char> workspace;

    float alpha = 1.0f;
    float beta = 0.0f;

    cusparseDnVecDescr_t srcDesc, dstDesc;
    cusparseSpMatDescr_t matDesc;

    CuSpmv(cusparseHandle_t handle, const linalg::CsrMatrix<float> & m)
        : handle(handle), src(m.cols), dst(m.rows)
    {
        mat.upload(m);

        // Create cusparse descriptors
        auto rc = cusparseCreateDnVec(&srcDesc, src.size(), src.get(), cudaDataType::CUDA_R_32F);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to source vector: {}", cusparseGetErrorName(rc)));
        }

        rc = cusparseCreateDnVec(&dstDesc, dst.size(), dst.get(), cudaDataType::CUDA_R_32F);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to source vector: {}", cusparseGetErrorName(rc)));
        }

        rc = cusparseCreateCsr(&matDesc, mat.rows, mat.cols,
                               mat.values.size(), mat.rowStart.get(),
                               mat.column.get(), mat.values.get(),
                               cusparseIndexType_t::CUSPARSE_INDEX_32I,
                               cusparseIndexType_t::CUSPARSE_INDEX_32I,
                               cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                               cudaDataType::CUDA_R_32F);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create CSR: {}", cusparseGetErrorName(rc)));
        }

        size_t spmvBufferSize = 0;
        rc = cusparseSpMV_bufferSize(handle, op,
                                     &alpha, matDesc, srcDesc,
                                     &beta, dstDesc,
                                     cudaDataType::CUDA_R_32F, cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                                     &spmvBufferSize);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseSpMV_bufferSize failed: {}", cusparseGetErrorName(rc)));
        }
        std::cout << "Workspace buffer size: " << spmvBufferSize << "\n";

        workspace = cu::vec<char>(spmvBufferSize);

        rc = cusparseSpMV_preprocess(handle, op,
                                     &alpha, matDesc, srcDesc, &beta, dstDesc,
                                     cudaDataType::CUDA_R_32F, cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                                     workspace.get());
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseSpMV_preprocess failed: {}", cusparseGetErrorName(rc)));
        }
    }

    void calculate()
    {
        auto rc = cusparseSpMV(handle, op, &alpha,
                               matDesc, srcDesc, &beta, dstDesc,
                               cudaDataType::CUDA_R_32F, cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                               workspace.get());
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseSpMV failed: {}", cusparseGetErrorName(rc)));
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

int main(int argc, char ** argv)
{
    const std::string usageMsg = "./SpmvTest <csr matrix>";
    if (argc != 2)
    {
        std::cerr << usageMsg << "\n";
        return 1;
    }

    // testMemorySpeed();

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

    cusparseHandle_t handle;
    auto rc = cusparseCreate(&handle);
    if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
    {
        std::cerr << "Failed to create cusparse: " << cusparseGetErrorName(rc) << "\n";
        return 1;
    }

    CuSpmv cuSpmv(handle, m);

    const int nRuns = 10;

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

    std::cout << "CUDA times:\n";
    sum = 0;
    for (int i = 0; i < nRuns; i++)
    {
        u::Stopwatch sw;
        cuSpmv.calculate();
        const auto currT = sw.millis();
        std::cout << currT << " ms\n";
        sum += currT;
    }    
    const double avgCuda = sum / nRuns;

    std::vector<float> resultCuda(nRows);
    cuSpmv.dst.download(resultCuda);
    double sumSq = 0;
    float maxDelta = 0;
    for (int i = 0; i < nRows; i++)
    {
        const float a = resultS[i];
        const float b = resultCuda[i];
        const float delta = a - b;
        sumSq += delta * delta;
        if (std::abs(delta) > std::abs(maxDelta))
        {
            maxDelta = delta;
        }
        if (std::abs(delta) > 1e-3f)
        {
            std::cout << i << ": " << delta << "\n";
        }
    }
    const double avg = std::sqrt(sumSq / nRows);
    std::cout << "Average error: " << avg << ", max delta = " << maxDelta << "\n";

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

    sumSq = 0;
    maxDelta = 0;
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