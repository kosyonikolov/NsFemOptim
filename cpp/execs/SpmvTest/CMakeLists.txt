add_executable(SpmvTest src/spmvTest.cpp)
target_include_directories(SpmvTest PRIVATE include/)
target_link_libraries(SpmvTest LinAlg Utils)