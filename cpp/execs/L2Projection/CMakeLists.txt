add_executable(L2Projection src/l2Projection.cpp)
target_link_libraries(L2Projection Mesh)

find_package(Eigen3 REQUIRED)
target_link_libraries(L2Projection Eigen3::Eigen)