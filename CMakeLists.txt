cmake_minimum_required(VERSION 3.10)

project(Lab6_PDE)

enable_testing()

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
add_executable(testLinearOutputNeuralNetwork tests/testLinearOutputNeuralNetwork.cpp src/linalg.cpp src/neuralNetworkFunctions.cpp src/neuralNetwork.cpp)


# Register tests
add_test(NAME LinalgVectorTest COMMAND ${CMAKE_SOURCE_DIR}/bin/testLinalgVector)
add_test(NAME LinalgMatrixTest COMMAND ${CMAKE_SOURCE_DIR}/bin/testLinalgMatrix)
add_test(NAME FemurTest        COMMAND ${CMAKE_SOURCE_DIR}/bin/testFemur)
add_test(NAME NNFunctionsTest  COMMAND ${CMAKE_SOURCE_DIR}/bin/testNNFunctions)
add_test(NAME NeuralNetworkTest COMMAND ${CMAKE_SOURCE_DIR}/bin/testNeuralNetwork)
add_test(NAME NNLoadTest       COMMAND ${CMAKE_SOURCE_DIR}/bin/testNeuralNetworkLoadConstructor)
add_test(NAME LinearNNTest     COMMAND ${CMAKE_SOURCE_DIR}/bin/testLinearOutputNeuralNetwork)

# Doxygen documentation target
find_package(Doxygen)
if(DOXYGEN_FOUND)
    add_custom_target(doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${CMAKE_SOURCE_DIR}/Doxyfile
        WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
        COMMENT "Generating documentation with Doxygen"
        VERBATIM
    )
    
    add_custom_target(doc-open
        COMMAND firefox ${CMAKE_SOURCE_DIR}/doc/html/index.html
        DEPENDS doc
        COMMENT "Opening documentation in Firefox"
        VERBATIM
    )
    
    message(STATUS "Doxygen found. Use 'make doc' to generate documentation")
    message(STATUS "Use 'make doc-open' to generate and open in Firefox")
else()
    message(WARNING "Doxygen not found. Documentation generation disabled")
endif()
add_executable(benchmarkNN tests/NeuralNetworkBenchmark.cpp src/linalg.cpp src/neuralNetwork.cpp src/neuralNetworkFunctions.cpp)
add_executable(testBinarySavePerformance tests/testBinarySavePerformance.cpp src/linalg.cpp src/neuralNetwork.cpp src/neuralNetworkFunctions.cpp)

# ============================================
# Python bindings with pybind11
# ============================================
# Allow user to specify Python3 executable (e.g., from venv)
set(Python3_EXECUTABLE "${Python3_EXECUTABLE}" CACHE FILEPATH "Python 3 interpreter")
find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
find_package(pybind11 CONFIG)

if(pybind11_FOUND)
    message(STATUS "pybind11 found - building Python module femur_rdn")
    
    pybind11_add_module(femur_rdn 
        src/pyBindings.cpp 
        src/linalg.cpp 
        src/neuralNetwork.cpp 
        src/neuralNetworkFunctions.cpp
    )
    
    target_include_directories(femur_rdn PRIVATE 
        ${CMAKE_SOURCE_DIR}/include
        ${CMAKE_SOURCE_DIR}/lib/eigen-5.0.0
        ${CMAKE_SOURCE_DIR}/lib/pybind11
    )
    
    # Output the .so file to visualization/ for easy import
    set_target_properties(femur_rdn PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/scripts/visualization/lib
    )
else()
    message(WARNING "pybind11 not found - Python bindings disabled. Install with: pip install pybind11")
endif()
