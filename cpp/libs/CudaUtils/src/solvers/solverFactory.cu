#include <cu/solvers/solverFactory.h>

#include <format>
#include <stdexcept>

#include <cu/gaussSeidel.h>
#include <cu/dssSolver.h>
#include <cu/sparse.h>
#include <cu/spmm.h>

namespace cu
{
    class GsWrapper : public AbstractSolver
    {
        Sparse sparse;
        Blas blas;
        std::unique_ptr<GaussSeidel> core;

        int numCh;
        int maxIters;
        float targetMse;

        float lastCombinedMse;

    public:
        GsWrapper(const int numCh, const linalg::CsrMatrix<float> & m,
                  const int maxIters, const float targetMse) : numCh(numCh), maxIters(maxIters), targetMse(targetMse)
        {
            core = std::make_unique<GaussSeidel>(blas, sparse.handle(), m, numCh);
            lastCombinedMse = -1;
        }

        int getNumCh() const
        {
            return numCh;
        }

        float getLastMse(const int ch) const
        {
            if (ch < 0)
            {
                return lastCombinedMse;
            }
            if (ch >= 0 && ch < core->lastMse.size())
            {
                return core->lastMse[ch];
            }
            return -1;
        }

        int getLastIterations() const
        {
            return core->lastIterations;
        }

        cu::vec<float> & getRhs()
        {
            return core->ioRhs;
        }

        cu::vec<float> & getSol()
        {
            return core->ioSol;
        }

        void solve()
        {
            lastCombinedMse = core->solve(maxIters, targetMse);
        }

        void setMaxIters(const int n)
        {
            maxIters = n;
        }

        void setTargetMse(const float mse)
        {
            targetMse = mse;
        }

        void setMseCheckInterval(const int mseMod)
        {
            core->setMseCheckInterval(mseMod);
        }
    };

    class DssWrapper : public AbstractSolver
    {
        Dss dss;
        std::unique_ptr<DssSolver> core;
        int numCh;

        // For MSE calculation
        bool calculateMse;
        std::unique_ptr<Blas> blas;
        std::unique_ptr<Sparse> sparse;
        std::unique_ptr<csrF> cudaMat;
        std::unique_ptr<spmm> mseSpmm;

        std::vector<float> lastMse;
        float lastCombinedMse = -1;

    public:
        DssWrapper(const int numCh, const linalg::CsrMatrix<float> & m,
                   const bool calculateMse) : numCh(numCh), calculateMse(calculateMse)
        {
            core = std::make_unique<DssSolver>(dss, m, numCh, cudssMatrixType_t::CUDSS_MTYPE_SPD);
            core->analyze();

            lastMse.resize(numCh, -1);

            if (calculateMse)
            {
                // Only initialize extra stuff if we are going to be calculating the MSE
                blas = std::make_unique<Blas>();
                sparse = std::make_unique<Sparse>();
                cudaMat = std::make_unique<csrF>(m);
                mseSpmm = std::make_unique<spmm>(sparse->handle(), *cudaMat, numCh);
            }
        }

        int getNumCh() const
        {
            return numCh;
        }

        float getLastMse(const int ch) const
        {
            if (ch < 0)
            {
                return lastCombinedMse;
            }
            if (ch > 0 && ch < lastMse.size())
            {
                return lastMse[ch];
            }
            return -1;
        }

        int getLastIterations() const
        {
            return 1;
        }

        cu::vec<float> & getRhs()
        {
            return core->rhs;
        }

        cu::vec<float> & getSol()
        {
            return core->sol;
        }

        void solve()
        {
            assert(core);
            core->solve();

            if (calculateMse)
            {
                assert(sparse);
                assert(blas);
                assert(mseSpmm);

                auto & sol = core->sol;
                auto & rhs = core->rhs;
                const int totalSize = sol.size();
                const int n = totalSize / numCh;
                assert(n * numCh == totalSize);
                assert(rhs.size() == sol.size());

                auto & resXy = mseSpmm->b;
                assert(resXy.size() == rhs.size());
                mseSpmm->compute(sol, resXy);
                cu::saxpy(*blas, totalSize, rhs.get(), resXy.get(), -1.0f);

                float sumNorm2 = 0;
                for (int c = 0; c < numCh; c++)
                {
                    float norm2 = -1; // == sqrt(sum(res[i]^2))
                    auto rc = cublasSnrm2(blas->handle, n, resXy.get() + c * n, 1, &norm2);
                    if (rc != cublasStatus_t::CUBLAS_STATUS_SUCCESS)
                    {
                        throw std::runtime_error(std::format("cublasSnrm2 failed: {}", cublasGetStatusName(rc)));
                    }

                    lastMse[c] = norm2 / std::sqrt(n);
                    sumNorm2 += norm2;
                }

                lastCombinedMse = sumNorm2 / std::sqrt(totalSize);
            }
        }

        void setMaxIters(const int)
        {
        }

        void setTargetMse(const float)
        {
        }

        void setMseCheckInterval(const int)
        {
        }
    };

    std::unique_ptr<AbstractSolver> createSolver(const std::string & name, const int numCh, const linalg::CsrMatrix<float> & m,
                                                 const int maxIters, const float targetMse, const int mseCheckInterval)
    {
        if (numCh < 1)
        {
            throw std::invalid_argument(std::format("Invalid number of channels for solver: {}", numCh));
        }
        if (mseCheckInterval < 1)
        {
            throw std::invalid_argument(std::format("Invalid MSE check interval for solver: {}", mseCheckInterval));
        }

        if (name == "gs")
        {
            auto res = std::make_unique<GsWrapper>(numCh, m, maxIters, targetMse);
            res->setMseCheckInterval(mseCheckInterval);
            return res;
        }
        else if (name == "dss")
        {
            // TODO Make this a CMake option
            constexpr bool dssWithMse = true;

            auto res = std::make_unique<DssWrapper>(numCh, m, dssWithMse);
            return res;
        }

        throw std::invalid_argument(std::format("Invalid solver name: {}", name));
    }
} // namespace cu
