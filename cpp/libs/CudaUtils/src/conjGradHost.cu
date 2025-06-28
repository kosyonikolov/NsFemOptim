#include <cu/conjGradHost.h>

#include <cu/conjGradF.h>

namespace cu
{
    struct CgCudaBuffs
    {
        cu::vec<float> rhs;
        cu::vec<float> sol;

        CgCudaBuffs(const int n) : rhs(n), sol(n)
        {

        }
    };

    ConjGradHost::ConjGradHost(const linalg::CsrMatrix<float> & m)
    {
        cudaMat = std::make_unique<csrF>(m);
        cudaCg = std::make_unique<ConjugateGradientF>(*cudaMat);
        buffs = std::make_unique<CgCudaBuffs>(m.rows);
    }

    ConjGradHost::~ConjGradHost()
    {
    }

    float ConjGradHost::solve(const std::vector<float> & rhs, std::vector<float> & sol,
                              const int maxIters, const float target)
    {
        assert(rhs.size() == sol.size());
        buffs->rhs.upload(rhs);
        buffs->sol.upload(sol);
        const float mse = cudaCg->solve(buffs->rhs, buffs->sol, maxIters, target);
        buffs->sol.download(sol);
        return mse;
    }
} // namespace cu