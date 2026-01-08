cmake_minimum_required(VERSION 3.10)

project(Lab6_PDE)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)


include_directories(include)
include_directories(${CMAKE_SOURCE_DIR}/lib/eigen-5.0.0)

set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)


### Main executable
add_executable(main src/main.cpp src/linalg.cpp src/femur.cpp src/neuralNetwork.cpp src/neuralNetworkFunctions.cpp)
### Tests
add_executable(testLinalgVector tests/testLinalgVector.cpp src/linalg.cpp)
add_executable(testLinalgMatrix tests/testLinalgMatrix.cpp src/linalg.cpp)
add_executable(testFemur tests/testFemur.cpp src/femur.cpp src/linalg.cpp)
add_executable(testNNFunctions tests/testNNFunctions.cpp src/linalg.cpp src/neuralNetworkFunctions.cpp)

add_executable(testNeuralNetwork tests/testNeuralNetwork.cpp src/linalg.cpp src/neuralNetworkFunctions.cpp src/neuralNetwork.cpp)
add_executable(testNeuralNetworkLoadConstructor tests/testNeuralNetworkLoadConstructor.cpp src/linalg.cpp src/neuralNetworkFunctions.cpp src/neuralNetwork.cpp)
