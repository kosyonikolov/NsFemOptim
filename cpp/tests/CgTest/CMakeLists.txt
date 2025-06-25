add_executable(CgTest src/cgTest.cu)
target_link_libraries(CgTest LinAlg CudaUtils)