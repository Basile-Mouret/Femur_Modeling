# PCA Implementation Documentation

## Overview

This document describes the Principal Component Analysis (PCA) implementation for statistical shape modeling of femur data.

## Files Created

### C++ Implementation

| File | Description |
|------|-------------|
| [include/pca.hpp](../include/pca.hpp) | PCA class template (header-only with full implementation) |
| [include/dataset.hpp](../include/dataset.hpp) | FemurDataset utility class for loading and managing shape data |
| [tests/testPCA.cpp](../tests/testPCA.cpp) | Comprehensive unit tests |

### Python Visualization

| File | Description |
|------|-------------|
| [visualization/pca_visualizer.py](../visualization/pca_visualizer.py) | Comprehensive PCA visualization tools |
| [visualization/pca_explorer.py](../visualization/pca_explorer.py) | Interactive GUI with sliders |
| [visualization/reconstruction_analysis.py](../visualization/reconstruction_analysis.py) | Reconstruction quality analysis |

## Class API Reference

### PCA Class (`PCA<T>`)

A template class for Principal Component Analysis, supporting `float` or `double` types.

#### Core Methods

```cpp
// Fitting
void fit(const Matrix2D<T>& data, int maxComponents = -1);

// Transformation
Vector<T> transform(const Vector<T>& shape, int numComponents = -1) const;
Vector<T> inverseTransform(const Vector<T>& coefficients) const;
Vector<T> reconstruct(const Vector<T>& shape, int numComponents = -1) const;
```

#### Analysis Methods

```cpp
T reconstructionError(const Vector<T>& shape, int numComponents = -1) const;
Vector<T> explainedVarianceRatio() const;
Vector<T> cumulativeVarianceRatio() const;
size_t componentsForVariance(T varianceThreshold) const;
```

#### Generation Methods

```cpp
Vector<T> generateShape(const Vector<T>& weights) const;
Vector<T> generateAlongMode(size_t mode, T sigma) const;
Matrix2D<T> sampleShapes(size_t numSamples, int numComponents = -1) const;
```

#### I/O Methods

```cpp
void save(const std::string& filename) const;
void load(const std::string& filename);
void printSummary() const;
```

### FemurDataset Class (`FemurDataset<T>`)

A utility class for loading and managing multiple femur shapes.

```cpp
// Loading
size_t loadFromDirectory(const std::string& directory, bool standardized = true, unsigned int sampleRate = 1);
void addFemur(const Femur& femur, const std::string& filename = "");

// Data access
Matrix2D<T> toMatrix() const;               // Returns D x N matrix
Vector<T> getShapeVector(size_t index) const;
const Femur& getFemur(size_t index) const;
Vector<T> femurToVector(const Femur& femur) const;

// Conversion
Femur vectorToFemur(const Vector<T>& vec, const Femur& templateFemur) const;
static Vector<T> loadShapeFromFile(const std::string& filename, bool standardized = true, unsigned int sampleRate = 1);

// Properties
size_t size() const;
size_t getNumVertices() const;
size_t getDimension() const;
bool isLoaded() const;
bool isStandardized() const;
unsigned int getSampleRate() const;
```

## Mathematical Background

### PCA Algorithm

Given a data matrix $X \in \mathbb{R}^{D \times N}$ with $N$ shapes in $D$ dimensions:

1. **Center the data**: $\bar{x} = \frac{1}{N} \sum_{i=1}^N x_i$, $\tilde{X} = X - \bar{x} \mathbf{1}^T$

2. **Compute SVD**: $\tilde{X} = U \Sigma V^T$ where:
   - $U \in \mathbb{R}^{D \times N}$ contains principal components
   - $\Sigma$ contains singular values
   - $V \in \mathbb{R}^{N \times N}$ contains right singular vectors

3. **Extract eigenvalues**: $\lambda_k = \frac{\sigma_k^2}{N-1}$

### Shape Model

A shape can be represented as:
$$s = \bar{s} + \sum_{k=1}^K \alpha_k \sqrt{\lambda_k} \cdot v_k$$

Where:
- $\bar{s}$ is the mean shape
- $v_k$ are principal components (modes of variation)
- $\lambda_k$ are variances
- $\alpha_k$ are shape parameters (typically $|\alpha_k| \leq 3$)

## Test Results

### Synthetic Data Tests

| Test | Status |
|------|--------|
| Synthetic 2D data | ✓ Passed |
| Transform consistency | ✓ Passed |
| Partial reconstruction | ✓ Passed |
| Shape generation | ✓ Passed |
| Save/Load | ✓ Passed |
| Variance analysis | ✓ Passed |

### Femur Data Integration

Training on 22 femur shapes (54,873 dimensions each):

| Metric | Value |
|--------|-------|
| Components for 90% variance | 7 |
| Components for 95% variance | 10 |
| Components for 99% variance | 17 |
| Total components | 21 (= N-1) |

**Variance Explained by First 10 Components:**

| Component | Individual | Cumulative |
|-----------|------------|------------|
| PC1 | 42.50% | 42.50% |
| PC2 | 26.81% | 69.31% |
| PC3 | 8.77% | 78.09% |
| PC4 | 5.86% | 83.95% |
| PC5 | 3.55% | 87.50% |
| PC6 | 2.24% | 89.74% |
| PC7 | 2.01% | 91.75% |
| PC8 | 1.64% | 93.39% |
| PC9 | 1.14% | 94.53% |
| PC10 | 1.10% | 95.63% |

## Usage Example

```cpp
#include "pca.hpp"
#include "dataset.hpp"

// Load dataset (standardized coordinates, sample rate 1)
FemurDataset<double> dataset("data/training/", true, 1);
Matrix2D<double> dataMatrix = dataset.toMatrix();

// Fit PCA
PCA<double> pca;
pca.fit(dataMatrix);
pca.printSummary();

// Transform a shape to latent space
Vector<double> shape = dataset.getShapeVector(0);
Vector<double> coeffs = pca.transform(shape, 10);  // Use first 10 components

// Reconstruct
Vector<double> reconstructed = pca.inverseTransform(coeffs);

// Generate shape variations
Vector<double> plusMode0 = pca.generateAlongMode(0, 2.0);   // +2σ along PC1
Vector<double> minusMode0 = pca.generateAlongMode(0, -2.0); // -2σ along PC1

// Save model
pca.save("bin/pca_model.bin");

// Export generated shape as OBJ (requires template for mesh connectivity)
const Femur& templateFemur = dataset.getFemur(0);
Femur generated = dataset.vectorToFemur(plusMode0, templateFemur);
generated.saveToFile("output/shape_pc1_plus2sigma.obj");
```

## Build Instructions

The PCA test is integrated into the CMake build system:

```bash
cd build
cmake ..
make testPCA
../bin/testPCA
```

## File Format

PCA models are saved in binary format with header "PCA1":

```
[4 bytes]  Magic: "PCA1"
[8 bytes]  numDimensions (size_t)
[8 bytes]  numSamples (size_t)
[8 bytes]  numComponents (size_t)
[8 bytes]  totalVariance (T)
[D × sizeof(T)] mean vector
[K × sizeof(T)] variances vector
[D × K × sizeof(T)] components matrix (column-major)
```

## Known Limitations

1. **Memory**: Full data matrix must fit in memory (54,873 × N doubles)
2. **Sample requirement**: At least 2 samples required for SVD
3. **Point correspondence**: Requires all shapes to have identical vertex ordering (provided in dataset)
4. **Linear model**: Cannot capture non-linear shape variations

## Python Visualization Usage

### Setup

The project uses a single virtual environment at the repository root (`.venv`).

```bash
# From repository root
cd /path/to/Femur_Modeling

# For fish shell
source .venv/bin/activate.fish

# For bash/zsh
source .venv/bin/activate
```

### Quick Examples

All commands should be run from the **repository root**.

```bash
# Show mean shape (interactive 3D)
python visualization/pca_visualizer.py \
    -m bin/pca_femur_model.bin \
    -t data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --mean

# Show first 5 modes in a grid
python visualization/pca_visualizer.py \
    -m bin/pca_femur_model.bin \
    -t data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --modes 5

# Animate PC1 oscillating between ±3σ
python visualization/pca_visualizer.py \
    -m bin/pca_femur_model.bin \
    -t data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --animate 0

# Interactive explorer with sliders
python visualization/pca_explorer.py \
    -m bin/pca_femur_model.bin \
    -t data/training/L_Femur_11_DECIM.obj.FINAL.obj

# Reconstruction analysis
python visualization/reconstruction_analysis.py \
    -m bin/pca_femur_model.bin \
    -t data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    -s data/validation/R_Femur_22_DECIM.obj.FINAL.obj
```

---

## Interactive Explorer Guide

The **PCA Explorer** provides an interactive GUI for real-time exploration of the shape model.

### Launch Command

```bash
python visualization/pca_explorer.py \
    --model bin/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --sliders 10 \
    --range 3.0
```

**Options:**
- `--sliders N`: Number of principal components to control (default: 10)
- `--range R`: Slider range in standard deviations (default: ±3.0σ)

### Controls

| Control | Action |
|---------|--------|
| **Sliders (left panel)** | Adjust principal component weights |
| **Left-click + drag** | Rotate the 3D view |
| **Right-click + drag** | Pan the view |
| **Scroll wheel** | Zoom in/out |
| **R key** | Reset camera to default view |
| **Q key** | Quit the application |

### Understanding the Sliders

Each slider controls one principal component (mode of variation). The value represents the number of standard deviations (σ) along that mode:

| Value | Meaning |
|-------|---------|
| **0** | Mean shape (no deformation) |
| **+1 to +3** | Shape deformed in positive direction |
| **-1 to -3** | Shape deformed in negative direction |

The percentage next to each PC label (e.g., "PC1 (42.5%)") indicates how much of the total variance that component explains.

### Shape Generation Formula

The displayed shape is computed as:

$$\mathbf{s} = \bar{\mathbf{s}} + \sum_{k=1}^{K} \alpha_k \sqrt{\lambda_k} \cdot \mathbf{v}_k$$

Where:
- $\bar{\mathbf{s}}$ is the mean shape
- $\alpha_k$ is the slider value for PC$k$ (in σ units)
- $\lambda_k$ is the variance of PC$k$
- $\mathbf{v}_k$ is the $k$-th principal component

### Typical Exploration Workflow

1. **Start at mean**: All sliders at 0 → observe the average femur shape
2. **Explore PC1**: Move slider 1 from -3 to +3 → see the dominant mode of variation
3. **Reset and explore PC2**: Return slider 1 to 0, adjust slider 2
4. **Combine modes**: Adjust multiple sliders to create novel shape combinations
5. **Note extremes**: Shapes beyond ±3σ are statistically unlikely

### Anatomical Interpretation

*TODO: Document anatomical interpretation of principal components after analysis.*

---

### Python API

```python
from pca_visualizer import load_pca_model, load_template_mesh, PCAVisualizer

# Load model and create visualizer
model = load_pca_model('bin/pca_femur_model.bin')
template = load_template_mesh('data/training/L_Femur_11.obj')
viz = PCAVisualizer(model, template)

# Visualize
viz.show_mean_shape()
viz.show_mode_variation(mode=0)
viz.animate_mode(mode=0)
viz.plot_variance_explained()

# Export
viz.export_mean_shape('mean.obj')
viz.export_mode_variations('output/', n_modes=5)
```

## Future Improvements


- [ ] Kernel PCA for non-linear extensions
- [ ] Probabilistic PCA (PPCA) for missing data
- [ ] Incremental PCA for online learning
- [ ] GPU acceleration via CUDA
