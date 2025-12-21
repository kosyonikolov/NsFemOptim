set(NS_SOURCES src/navierStokes.cpp
               src/nsConfig.cpp
               src/chorinEigen.cpp
               src/buildContext.cpp
               src/log.cpp
               src/progressTracker.cpp
               src/basicOutputHandler.cpp)

if (CMAKE_CUDA_COMPILER)
    list(APPEND NS_SOURCES src/chorinCuda.cu)
else()
    list(APPEND NS_SOURCES src/chorinCudaFallback.cpp)
endif()

add_executable(NavierStokes ${NS_SOURCES})
target_link_libraries(NavierStokes Mesh Utils LinAlg Fem)
target_include_directories(NavierStokes PRIVATE include/)

find_package(Eigen3 REQUIRED)
target_link_libraries(NavierStokes Eigen3::Eigen)

if (CMAKE_CUDA_COMPILER)
    target_link_libraries(NavierStokes CudaUtils)
endif()