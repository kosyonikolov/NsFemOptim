add_executable(ConvergenceAnalyzer src/convergenceAnalyzer.cpp
                                   src/config.cpp)
target_link_libraries(ConvergenceAnalyzer Utils LinAlg)
target_include_directories(ConvergenceAnalyzer PRIVATE include/)
