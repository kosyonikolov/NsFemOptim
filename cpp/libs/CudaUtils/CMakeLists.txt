add_library(CudaUtils src/cusparse.cu src/csrF.cu)
target_include_directories(CudaUtils PUBLIC include/)
target_link_libraries(CudaUtils PUBLIC Utils LinAlg)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(CudaUtils PUBLIC CUDA::cusparse)
