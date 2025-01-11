add_library(Mesh src/gmsh.cpp src/io.cpp
                 src/element.cpp src/affineTransform.cpp
                 src/concreteMesh.cpp)
target_include_directories(Mesh PUBLIC include/)