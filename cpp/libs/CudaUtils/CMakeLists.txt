add_library(CudaUtils src/sparse.cu src/vec.cu src/csrF.cu src/spmv.cu
                      src/blas.cu src/conjGradF.cu
                      src/dss.cu src/dssSolver.cu
                      src/gaussSeidel.cu
                      src/gaussSeidelHost.cu
                      src/stopwatch.cu
                      src/conjGradHost.cu
                      src/spmm.cu
                      src/solvers/solverFactory.cu)
target_include_directories(CudaUtils PUBLIC include/)
target_link_libraries(CudaUtils PUBLIC Utils LinAlg)

find_package(CUDAToolkit REQUIRED)
find_package(cudss REQUIRED)
target_link_libraries(CudaUtils PUBLIC CUDA::cublas CUDA::cusparse cudss)

option(CU_SOLVERS_ITER_LOG "Log CUDA iteration steps of CUDA solvers" OFF)
if (CU_SOLVERS_ITER_LOG)
    target_compile_definitions(CudaUtils PRIVATE -DCU_SOLVERS_ITER_LOG)
endif()