#ifndef LIBS_CUDAUTILS_INCLUDE_CU_DATATYPES
#define LIBS_CUDAUTILS_INCLUDE_CU_DATATYPES

#include <cusparse.h>

namespace cu
{
    template <typename F>
    cudaDataType getCudaDataType();

    template <>
    inline cudaDataType getCudaDataType<float>()
    {
        return CUDA_R_32F;
    }

    template <>
    inline cudaDataType getCudaDataType<double>()
    {
        return CUDA_R_64F;
    }
} // namespace cu

#endif /* LIBS_CUDAUTILS_INCLUDE_CU_DATATYPES */
