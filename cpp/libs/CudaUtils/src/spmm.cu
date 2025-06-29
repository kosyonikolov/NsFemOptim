#include <cu/spmm.h>

namespace cu
{
    spmm::spmm(cusparseHandle_t handle, cu::csrF & m,
               const int numCh)
        : handle(handle), mat(m), numCh(numCh),
          x(m.cols * numCh), b(m.rows * numCh)
    {
        if (numCh < 1)
        {
            throw std::invalid_argument("SPMM channels must be at least 1");
        }

        matDesc = mat.getCuSparseDescriptor();

        auto xDesc = x.getCuSparseMatDescriptor(numCh);
        auto bDesc = b.getCuSparseMatDescriptor(numCh);

        size_t spmmBufferSize = 0;
        auto rc = cusparseSpMM_bufferSize(handle, op, op,
                                          &alpha, matDesc, xDesc,
                                          &beta, bDesc,
                                          cudaDataType::CUDA_R_32F, cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
                                          &spmmBufferSize);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseSpMM_bufferSize failed: {}", cusparseGetErrorName(rc)));
        }
        std::cout << "Workspace buffer size: " << spmmBufferSize << "\n";

        workspace = cu::vec<char>(spmmBufferSize);

        rc = cusparseSpMM_preprocess(handle, op, op,
                                     &alpha, matDesc, xDesc, &beta, bDesc,
                                     cudaDataType::CUDA_R_32F, cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
                                     workspace.get());
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseSpMV_preprocess failed: {}", cusparseGetErrorName(rc)));
        }
    }

    spmm::~spmm()
    {
    }

    void spmm::compute()
    {
        return compute(x.getCuSparseMatDescriptor(numCh), b.getCuSparseMatDescriptor(numCh));
    }

    void spmm::compute(cusparseDnMatDescr_t otherX, cusparseDnMatDescr_t otherB)
    {
        auto rc = cusparseSpMM(handle, op, op,
                               &alpha, matDesc, otherX, &beta, otherB,
                               cudaDataType::CUDA_R_32F, cusparseSpMMAlg_t::CUSPARSE_SPMM_ALG_DEFAULT,
                               workspace.get());
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("cusparseSpMM failed: {}", cusparseGetErrorName(rc)));
        }
    }

    void spmm::compute(cu::vec<float> & otherX, cu::vec<float> & otherB)
    {
        auto xd = otherX.getCuSparseMatDescriptor(numCh);
        auto bd = otherB.getCuSparseMatDescriptor(numCh);
        compute(xd, bd);
    }

} // namespace cu