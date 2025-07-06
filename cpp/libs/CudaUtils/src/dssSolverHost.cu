#include <cu/dssSolverHost.h>

#include <cu/dssSolver.h>

namespace cu
{
    DssSolverHost::DssSolverHost(const linalg::CsrMatrix<float> & m, const int numCh)
    {
        dss = std::make_unique<Dss>();
        core = std::make_unique<DssSolver>(*dss, m, numCh, cudssMatrixType_t::CUDSS_MTYPE_SPD);
        core->analyze();
    }

    DssSolverHost::~DssSolverHost()
    {

    }

    void DssSolverHost::solve(const std::vector<float> & rhs, std::vector<float> & sol)
    {
        core->rhs.upload(rhs);
        core->solve();
        core->sol.download(sol);
    }
}