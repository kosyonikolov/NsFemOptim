add_executable(NavierStokes src/navierStokes.cpp
                            src/nsConfig.cpp
                            src/borders.cpp
                            src/chorinEigen.cpp)
target_link_libraries(NavierStokes Mesh Utils LinAlg Fem)
target_include_directories(NavierStokes PRIVATE include/)

find_package(Eigen3 REQUIRED)
target_link_libraries(NavierStokes Eigen3::Eigen)