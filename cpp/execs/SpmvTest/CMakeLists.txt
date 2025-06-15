add_executable(SpmvTest src/spmvTest.cu)
target_include_directories(SpmvTest PRIVATE include/)
target_link_libraries(SpmvTest LinAlg Utils CudaUtils)

find_package(CUDAToolkit REQUIRED)
target_link_libraries(SpmvTest CUDA::cusparse)
