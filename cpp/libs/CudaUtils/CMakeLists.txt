add_library(CudaUtils INTERFACE)
target_include_directories(CudaUtils INTERFACE include/)
target_link_libraries(CudaUtils INTERFACE Utils LinAlg)