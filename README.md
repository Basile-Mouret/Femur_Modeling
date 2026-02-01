# Femur Modeling

Statistical shape analysis of femur bones using PCA, neural networks and diffeomorphic registration (LDDMM).

## Overview

This project implements a pipeline for 3D femur shape modeling:
- **Custom C++ neural network** library with linear algebra operations (using Eigen)
- **PCA-based dimensionality reduction** for shape representation
- **LDDMM registration** for computing diffeomorphic mappings between femur shapes
- **Python bindings** via pybind11 for integration with visualization tools

## Building the C++ Code

### Requirements

- CMake >= 3.10
- C++17 compiler
- Eigen (included in `lib/eigen-5.0.0/`)

### Build Instructions

```bash
mkdir -p build && cd build
cmake ..
make
```

Executables are generated in the `bin/` directory.

### Running Tests

```bash
cd build
ctest
```

### Documentation (optional)

```bash
make doc        # Generate Doxygen documentation
```

## Additional Documentation

- **LDDMM Pipeline**: See [lddmm/README.md](lddmm/README.md)
- **Visualization Tools**: See [visualization/](visualization/)

