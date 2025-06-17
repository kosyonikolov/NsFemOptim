#include <cu/csrF.h>

#include <cu/cusparse.h>

namespace cu
{
    csrF::csrF(const linalg::CsrMatrix<float> & cpuMat)
        : rows(cpuMat.rows), cols(cpuMat.cols),
          values(cpuMat.values), column(cpuMat.column), rowStart(cpuMat.rowStart),
          x(cpuMat.cols), b(cpuMat.rows)
    {
        handle = getCuSparseHandle(); // TODO Support custom handles

        // Create cusparse descriptors
        auto rc = cusparseCreateDnVec(&xDesc, x.size(), x.get(), cudaDataType::CUDA_R_32F);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create x vector descriptor: {}", cusparseGetErrorName(rc)));
        }

        rc = cusparseCreateDnVec(&bDesc, b.size(), b.get(), cudaDataType::CUDA_R_32F);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create b vector descriptor: {}", cusparseGetErrorName(rc)));
        }

        rc = cusparseCreateCsr(&matDesc, rows, cols,
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

        size_t spmvBufferSize = 0;
        rc = cusparseSpMV_bufferSize(handle, op,
                                     &alpha, matDesc, xDesc,
                                     &beta, bDesc,
                                     cudaDataType::CUDA_R_32F, cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                                     &spmvBufferSize);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseSpMV_bufferSize failed: {}", cusparseGetErrorName(rc)));
        }
        std::cout << "Workspace buffer size: " << spmvBufferSize << "\n";

        workspace = cu::vec<char>(spmvBufferSize);

        rc = cusparseSpMV_preprocess(handle, op,
                                     &alpha, matDesc, xDesc, &beta, bDesc,
                                     cudaDataType::CUDA_R_32F, cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                                     workspace.get());
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseSpMV_preprocess failed: {}", cusparseGetErrorName(rc)));
        }
    }

    csrF::~csrF()
    {
        // TODO
    }

    void csrF::spmv()
    {
        return spmv(xDesc, bDesc);
    }

    void csrF::spmv(cusparseDnVecDescr_t otherX, cusparseDnVecDescr_t otherB)
    {
        auto rc = cusparseSpMV(handle, op, &alpha,
                               matDesc, otherX, &beta, otherB,
                               cudaDataType::CUDA_R_32F, cusparseSpMVAlg_t::CUSPARSE_SPMV_ALG_DEFAULT,
                               workspace.get());
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseSpMV failed: {}", cusparseGetErrorName(rc)));
        }
    }
} // namespace cu