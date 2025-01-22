add_library(Element src/element.cpp src/affineTransform.cpp
                    src/calc.cpp
                    src/triangleIntegrator.cpp
                    src/point.cpp)
target_include_directories(Element PUBLIC include/)

find_package(OpenCV REQUIRED)
target_link_libraries(Element ${OpenCV_LIBS})