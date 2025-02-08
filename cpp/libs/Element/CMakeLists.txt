add_library(Element src/element.cpp src/affineTransform.cpp
                    src/triangleIntegrator.cpp
                    src/point.cpp
                    src/p0.cpp
                    src/p1.cpp
                    src/p2.cpp
                    src/factory.cpp)
target_include_directories(Element PUBLIC include/)

find_package(OpenCV REQUIRED)
target_link_libraries(Element ${OpenCV_LIBS})