cmake_minimum_required(VERSION 3.10)

project(Lab6_PDE)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/lib/eigen-5.0.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

add_executable(main src/main.cpp)