add_library(Mesh src/gmsh.cpp src/io.cpp
                 src/element.cpp src/affineTransform.cpp
                 src/concreteMesh.cpp
                 src/drawMesh.cpp)
target_include_directories(Mesh PUBLIC include/)

find_package(OpenCV REQUIRED)
target_link_libraries(Mesh PUBLIC ${OpenCV_LIBS})
target_include_directories(Mesh PUBLIC ${OpenCV_INCLUDES})