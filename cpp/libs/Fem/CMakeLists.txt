add_library(Fem src/chorinMatrices.cpp src/chorinCsr.cpp
                src/fastConvection.cpp)
target_include_directories(Fem PUBLIC include/)
target_link_libraries(Fem PUBLIC LinAlg Mesh Element)
target_link_libraries(Fem PRIVATE Utils)

find_package(Eigen3 REQUIRED)
target_link_libraries(Fem PUBLIC Eigen3::Eigen)