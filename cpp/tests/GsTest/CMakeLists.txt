add_executable(GsTest src/gsTest.cpp)
target_include_directories(GsTest PRIVATE include/)
target_link_libraries(GsTest LinAlg Utils)