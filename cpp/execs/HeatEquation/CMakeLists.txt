add_executable(HeatEquation src/heatEquation.cpp)
target_link_libraries(HeatEquation Mesh)

find_package(Eigen3 REQUIRED)
target_link_libraries(HeatEquation Eigen3::Eigen)