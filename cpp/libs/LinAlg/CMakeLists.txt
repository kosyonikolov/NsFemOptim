add_library(LinAlg src/csrMatrix.cpp
                   src/eigen.cpp
                   src/gaussSeidel.cpp
                   src/io.cpp
                   src/graphs.cpp
                   src/sparseBuilder.cpp
                   src/sparseDokBuilder.cpp)
target_include_directories(LinAlg PUBLIC include/)

find_package(Eigen3 REQUIRED)
target_link_libraries(LinAlg Eigen3::Eigen)