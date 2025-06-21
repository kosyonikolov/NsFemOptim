#include <cu/spmv.h>

namespace cu
{
    spmv::spmv(cusparseHandle_t handle, cu::csrF & m)
        : handle(handle), mat(m), x(m.cols), b(m.rows)
    {
        auto xDesc = x.getCuSparseDescriptor();
        auto bDesc = b.getCuSparseDescriptor();
        matDesc = mat.getCuSparseDescriptor();

        size_t spmvBufferSize = 0;
        auto rc = cusparseSpMV_bufferSize(handle, op,
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

    spmv::~spmv()
    {
        
    }

    void spmv::compute()
    {
        return compute(x.getCuSparseDescriptor(), b.getCuSparseDescriptor());
    }

    void spmv::compute(cusparseDnVecDescr_t otherX, cusparseDnVecDescr_t otherB)
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

    void spmv::compute(cu::vec<float> & otherX, cu::vec<float> & otherB)
    {
        auto xd = otherX.getCuSparseDescriptor();
        auto bd = otherB.getCuSparseDescriptor();
        compute(xd, bd);
    }

} // namespace cu