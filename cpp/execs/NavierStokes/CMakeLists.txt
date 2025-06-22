add_executable(NavierStokes src/navierStokes.cpp
                            src/nsConfig.cpp
                            src/borders.cpp
                            src/chorinEigen.cpp
                            src/buildContext.cpp
                            src/chorinCuda.cu)
target_link_libraries(NavierStokes Mesh Utils LinAlg Fem CudaUtils)
target_include_directories(NavierStokes PRIVATE include/)

find_package(Eigen3 REQUIRED)
target_link_libraries(NavierStokes Eigen3::Eigen)