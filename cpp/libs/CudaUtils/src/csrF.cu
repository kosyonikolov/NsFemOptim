#include <cu/csrF.h>

#include <cu/cusparse.h>

namespace cu
{
    csrF::csrF(const linalg::CsrMatrix<float> & cpuMat)
        : rows(cpuMat.rows), cols(cpuMat.cols),
          values(cpuMat.values), column(cpuMat.column), rowStart(cpuMat.rowStart)
    {
    }

    csrF::~csrF()
    {
        if (matDesc)
        {
            auto rc = cusparseDestroySpMat(matDesc);
            assert(rc == cusparseStatus_t::CUSPARSE_STATUS_SUCCESS);
        }
    }

    cusparseSpMatDescr_t csrF::getCuSparseDescriptor()
    {
        if (matDesc != 0)
        {
            return matDesc;
        }

        auto rc = cusparseCreateCsr(&matDesc, rows, cols,
                                    values.size(), rowStart.get(),
                                    column.get(), values.get(),
                                    cusparseIndexType_t::CUSPARSE_INDEX_32I,
                                    cusparseIndexType_t::CUSPARSE_INDEX_32I,
                                    cusparseIndexBase_t::CUSPARSE_INDEX_BASE_ZERO,
                                    cudaDataType::CUDA_R_32F);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create CSR: {}", cusparseGetErrorName(rc)));
        }

        return matDesc;
    }
} // namespace cu