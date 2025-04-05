add_library(Fem src/chorinMatrices.cpp src/chorinCsr.cpp)
target_include_directories(Fem PUBLIC include/)
target_link_libraries(Fem PUBLIC LinAlg Mesh)