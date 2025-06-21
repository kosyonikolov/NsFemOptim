add_executable(CgTest src/cgTest.cpp)
target_link_libraries(CgTest LinAlg)

find_package(Eigen3 REQUIRED)
target_link_libraries(CgTest Eigen3::Eigen)