add_executable(CudaUtilsTest src/cudaUtilsTest.cu
                             src/uploadTest.cu)
target_link_libraries(CudaUtilsTest CudaUtils)

#set_source_files_properties(src/cudaUtilsTest.cpp PROPERTIES LANGUAGE CUDA)