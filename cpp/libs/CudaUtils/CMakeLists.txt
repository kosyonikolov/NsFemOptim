add_library(CudaUtils src/sparse.cu src/vec.cu src/csrF.cu src/spmv.cu
                      src/blas.cu src/conjGradF.cu)
target_include_directories(CudaUtils PUBLIC include/)
target_link_libraries(CudaUtils PUBLIC Utils LinAlg)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(CudaUtils PUBLIC CUDA::cublas CUDA::cusparse)
