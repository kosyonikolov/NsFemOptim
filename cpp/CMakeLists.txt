cmake_minimum_required(VERSION 3.25.0)
project(NsFemOptim VERSION 0.1.0 LANGUAGES C CXX)
include(CheckLanguage)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wno-sign-compare")

option(USE_SANITIZERS "Use ASan + UBSan" OFF)
if (USE_SANITIZERS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsanitize=address,undefined")
endif()

check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
    enable_language(CUDA)
    message(STATUS "CUDA found: ${CMAKE_CUDA_COMPILER}")
    set(CMAKE_CUDA_ARCHITECTURES native)
else()
    message("CUDA not found, some targets won't be built")
endif()

add_subdirectory(libs)
add_subdirectory(execs)
add_subdirectory(tests)