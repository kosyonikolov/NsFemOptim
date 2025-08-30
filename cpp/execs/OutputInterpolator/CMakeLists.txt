add_executable(OutputInterpolator src/outputInterpolator.cpp
                                  src/config.cpp)
target_link_libraries(OutputInterpolator Mesh Utils LinAlg Fem)
target_include_directories(OutputInterpolator PRIVATE include/)
