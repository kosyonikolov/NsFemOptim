#include <cu/vec.h>

#include <cu/datatypes.h>

namespace cu
{
    template<typename F>
    cusparseDnVecDescr_t vec<F>::getCuSparseDescriptor()
    {
        if (cuSparseDescriptor != 0)
        {
            return cuSparseDescriptor;
        }

        const auto dtype = getCudaDataType<F>();
        auto rc = cusparseCreateDnVec(&cuSparseDescriptor, size(), get(), dtype);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create vector descriptor: {}", cusparseGetErrorName(rc)));
        }

        return cuSparseDescriptor;
    }

    template<typename F>
    cusparseDnMatDescr_t vec<F>::getCuSparseMatDescriptor(const int numCh)
    {
        const auto n = size();
        if (numCh < 1 || n % numCh != 0)
        {
            throw std::invalid_argument(std::format("Bad number of channels [{}] for vector of size {}", numCh, n));
        }

        if (cuSparseMatDescriptor != 0)
        {
            return cuSparseMatDescriptor;
        }

        const auto dtype = getCudaDataType<F>();
        const size_t rows = n / numCh;
        const size_t cols = numCh;
        auto rc = cusparseCreateDnMat(&cuSparseMatDescriptor, 
                                      rows, cols, rows, 
                                      get(), dtype, cusparseOrder_t::CUSPARSE_ORDER_COL);
        if (rc != cusparseStatus_t::CUSPARSE_STATUS_SUCCESS)
        {
            throw std::runtime_error(std::format("Failed to create dense mat descriptor: {}", cusparseGetErrorName(rc)));
        }

        return cuSparseMatDescriptor;
    }

    template cusparseDnVecDescr_t vec<float>::getCuSparseDescriptor();
    template cusparseDnVecDescr_t vec<double>::getCuSparseDescriptor();

    template cusparseDnMatDescr_t vec<float>::getCuSparseMatDescriptor(const int numCh);
    template cusparseDnMatDescr_t vec<double>::getCuSparseMatDescriptor(const int numCh);
}