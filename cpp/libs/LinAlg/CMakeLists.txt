add_library(LinAlg src/csrMatrix.cpp
                   src/eigen.cpp)
target_include_directories(LinAlg PUBLIC include/)

find_package(Eigen3 REQUIRED)
target_link_libraries(LinAlg Eigen3::Eigen)