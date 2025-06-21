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

    template cusparseDnVecDescr_t vec<float>::getCuSparseDescriptor();
    template cusparseDnVecDescr_t vec<double>::getCuSparseDescriptor();
}