add_library(Mesh src/gmsh.cpp src/io.cpp
                 src/concreteMesh.cpp
                 src/drawMesh.cpp
                 src/interpolator.cpp
                 src/colorScale.cpp
                 src/triangleLookup.cpp)
target_include_directories(Mesh PUBLIC include/)
target_link_libraries(Mesh PUBLIC Element)

find_package(OpenCV REQUIRED)
target_link_libraries(Mesh PUBLIC ${OpenCV_LIBS})
target_include_directories(Mesh PUBLIC ${OpenCV_INCLUDES})