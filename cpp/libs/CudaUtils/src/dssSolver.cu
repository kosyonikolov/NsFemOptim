#include <cu/dssSolver.h>

#include <stdexcept>

namespace cu
{
    DssSolver::DssSolver(Dss & lib, const linalg::CsrMatrix<float> & m,
                         const int numRhs, const cudssMatrixType_t matrixType)
        : lib(lib), n(m.rows), numRhs(numRhs),
          values(m.values), column(m.column), rowStart(m.rowStart),
          rhs(numRhs * m.rows), sol(numRhs * m.cols)
    {
        if (m.rows != m.cols)
        {
            throw std::invalid_argument("DssSolver currently only supports square matrices");
        }

        auto rc = cudssConfigCreate(&solverConfig);
        if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cudssConfigCreate failed: {}", dssStatusName(rc)));
        }

        rc = cudssDataCreate(lib.handle(), &solverData);
        if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cudssDataCreate failed: {}", dssStatusName(rc)));
        }

        rc = cudssMatrixCreateDn(&bMat,
                                 n, numRhs, n,
                                 rhs.get(), cudaDataType::CUDA_R_32F,
                                 cudssLayout_t::CUDSS_LAYOUT_COL_MAJOR);
        if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cudssMatrixCreateDn failed for rhs: {}", dssStatusName(rc)));
        }
        rc = cudssMatrixCreateDn(&xMat,
                                 n, numRhs, n,
                                 sol.get(), cudaDataType::CUDA_R_32F,
                                 cudssLayout_t::CUDSS_LAYOUT_COL_MAJOR);
        if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cudssMatrixCreateDn failed for sol: {}", dssStatusName(rc)));
        }

        rc = cudssMatrixCreateCsr(&A, n, n, values.size(),
                                  rowStart.get(), 0,
                                  column.get(),
                                  values.get(),
                                  cudaDataType_t::CUDA_R_32I,
                                  cudaDataType_t::CUDA_R_32F,
                                  matrixType,
                                  cudssMatrixViewType_t::CUDSS_MVIEW_FULL,
                                  cudssIndexBase_t::CUDSS_BASE_ZERO);
        if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cudssMatrixCreateCsr failed: {}", dssStatusName(rc)));
        }
    }

    void DssSolver::analyze()
    {
        auto rc = cudssExecute(lib.handle(), CUDSS_PHASE_ANALYSIS, solverConfig, solverData, A, xMat, bMat);
        if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("CUDSS_PHASE_ANALYSIS failed: {}", dssStatusName(rc)));
        }

        rc = cudssExecute(lib.handle(), CUDSS_PHASE_FACTORIZATION, solverConfig, solverData, A, xMat, bMat);
        if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("CUDSS_PHASE_FACTORIZATION failed: {}", dssStatusName(rc)));
        }
        hasAnalyzed = true;
    }

    void DssSolver::solve()
    {
        if (!hasAnalyzed)
        {
            analyze();
        }

        auto rc = cudssExecute(lib.handle(), CUDSS_PHASE_SOLVE, solverConfig, solverData, A, xMat, bMat);
        if (rc != cudssStatus_t::CUDSS_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("CUDSS_PHASE_SOLVE failed: {}", dssStatusName(rc)));
        }
    }
} // namespace cu