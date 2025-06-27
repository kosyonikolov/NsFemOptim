#include <cu/gaussSeidelHost.h>

#include <cu/blas.h>
#include <cu/gaussSeidel.h>
#include <cu/sparse.h>

namespace cu
{
    GaussSeidelHost::GaussSeidelHost(const linalg::CsrMatrix<float> & m)
    {
        blas = std::make_unique<Blas>();
        sparse = std::make_unique<Sparse>();
        cudaGs = std::make_unique<GaussSeidel>(*blas, sparse->handle(), m);
    }

    GaussSeidelHost::~GaussSeidelHost()
    {
    }

    float GaussSeidelHost::solve(const std::vector<float> & rhs, std::vector<float> & sol,
                                 const int maxIters, const float target)
    {
        assert(rhs.size() == sol.size());
        cudaGs->rhs.upload(rhs);
        cudaGs->sol.upload(sol);
        const float mse = cudaGs->solve(maxIters, target);
        cudaGs->sol.download(sol);
        return mse;
    }
} // namespace cu