cmake_minimum_required(VERSION 3.5.0)
project(NsFemOptim VERSION 0.1.0 LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 20)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra ")

add_subdirectory(execs)