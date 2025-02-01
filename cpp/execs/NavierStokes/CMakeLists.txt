add_executable(NavierStokes src/navierStokes.cpp)
target_link_libraries(NavierStokes Mesh)

find_package(Eigen3 REQUIRED)
target_link_libraries(NavierStokes Eigen3::Eigen)