#ifndef LIBS_CUDAUTILS_INCLUDE_CU_DSSSOLVERHOST
#define LIBS_CUDAUTILS_INCLUDE_CU_DSSSOLVERHOST

#include <memory>

#include <linalg/csrMatrix.h>

namespace cu
{
    class Dss;
    class DssSolver;

    class DssSolverHost
    {
        std::unique_ptr<Dss> dss;
        std::unique_ptr<DssSolver> core;

    public:
        DssSolverHost(const linalg::CsrMatrix<float> & m, const int numCh = 1);

        ~DssSolverHost();

        void solve(const std::vector<float> & rhs, std::vector<float> & sol);
    };
} // namespace cu



#endif /* LIBS_CUDAUTILS_INCLUDE_CU_DSSSOLVERHOST */
