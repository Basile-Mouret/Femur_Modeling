# Femur Modeling Project - Architecture Plan

## Project Goals

Build a statistical shape analysis system for femur bones that enables:
1. **Mean shape computation** from a population
2. **Variance/mode analysis** to understand shape variability
3. **Dimensionality reduction** for compact representation
4. **Shape reconstruction** from partial data or constraints
5. **Shape synthesis** of new realistic femur shapes

---

## Implementation Progress

| Phase | Status | Files | Tests |
|-------|--------|-------|-------|
| Phase 1: PCA | ✅ Complete | `pca.hpp`, `dataset.hpp` | ✅ All pass |
| Phase 2: Autoencoder | ⏳ Pending | - | - |
| Phase 3: VAE | ⏳ Pending | - | - |
| Phase 4: LDDMM | ⏳ Pending | - | - |

---

## Implementation Phases

### Phase 1: Linear PCA (Foundation) ✅ COMPLETE

**Goal**: Implement classical Statistical Shape Model using PCA

**Status**: ✅ Implemented and tested

**Files Created**:
- [include/pca.hpp](../include/pca.hpp) - PCA template class
- [include/dataset.hpp](../include/dataset.hpp) - FemurDataset utility
- [tests/testPCA.cpp](../tests/testPCA.cpp) - Unit tests
- [docs/PCA_Documentation.md](PCA_Documentation.md) - Documentation

**API Implemented**:

```cpp
class PCA<T>
├── fit(data_matrix, maxComponents)      ✅
├── transform(shape, numComponents)       ✅
├── inverseTransform(coefficients)        ✅
├── reconstruct(shape, numComponents)     ✅
├── getMean()                             ✅
├── getComponents()                       ✅
├── getVariances()                        ✅
├── explainedVarianceRatio()              ✅
├── cumulativeVarianceRatio()             ✅
├── componentsForVariance(threshold)      ✅
├── generateShape(weights)                ✅
├── generateAlongMode(mode, sigma)        ✅
├── sampleShapes(numSamples, numComp)     ✅
├── save(filename)                        ✅
├── load(filename)                        ✅
└── printSummary()                        ✅

class FemurDataset<T>
├── loadFromDirectory(path, standardized, sampleRate)  ✅
├── addFemur(femur, filename)                          ✅
├── toMatrix()                                          ✅
├── getShapeVector(index)                               ✅
├── femurToVector(femur)                                ✅
├── vectorToFemur(vec, templateFemur)                   ✅
├── loadShapeFromFile(filename, standardized, sampleRate) [static] ✅
├── size(), getNumVertices(), getDimension()            ✅
└── printInfo()                                         ✅
```

**Test Results** (22 training femurs, D=54,873):
- Components for 90% variance: **7**
- Components for 95% variance: **10**  
- Components for 99% variance: **17**

**Python Visualization** ✅ Complete:
- `pca_visualizer.py` - Core visualization module
- `pca_explorer.py` - Interactive GUI with sliders
- `reconstruction_analysis.py` - Reconstruction quality analysis

---

### Phase 2: Autoencoder (Neural Nonlinear) ⏳ Priority: MEDIUM

**Goal**: Extend existing neural network to build shape autoencoders

**Architecture Options**:

```
Option A: Simple Autoencoder
[54873] ─> [1024] ─> [256] ─> [64] ─> [256] ─> [1024] ─> [54873]
  Input      Hidden layers      Latent    Hidden layers     Output

Option B: Deeper Autoencoder  
[54873] ─> [4096] ─> [1024] ─> [256] ─> [64] ─> [256] ─> [1024] ─> [4096] ─> [54873]
```

**Components to Build**:

```
src/autoencoder.cpp + include/autoencoder.hpp
├── class Autoencoder : extends NeuralNetwork
│   ├── Autoencoder(layers, activation, latent_dim)
│   ├── encode(shape)              // Get latent representation
│   ├── decode(latent)             // Reconstruct from latent
│   ├── train(dataset, epochs, batch_size)
│   ├── get_reconstruction_error()
│   ├── interpolate(shape1, shape2, alpha) // Latent space interpolation
│   └── sample(n_samples)          // Random latent → shape
```

**Training Considerations**:
- Small dataset (24 samples) → need regularization
- Consider data augmentation (small random perturbations)
- Use validation set (e.g., 4-5 femurs) for early stopping
- MSE loss on vertex coordinates

**Deliverables**:
- [ ] `Autoencoder` class
- [ ] Training script with loss curves
- [ ] Compare reconstruction error: PCA (K modes) vs AE (K latent dims)
- [ ] Shape interpolation visualization

---

### Phase 3: Variational Autoencoder (Generative) ⚠️ Priority: MEDIUM-LOW

**Goal**: Add probabilistic structure for better generative capability

**Architecture Changes**:
```
Encoder: Shape ─> [layers] ─> μ, log(σ²)  (two output heads)
Sampling: z = μ + σ × ε,  where ε ~ N(0, I)
Decoder: z ─> [layers] ─> Reconstructed Shape
```

**Loss Function**:
```cpp
double vae_loss = reconstruction_loss + beta * kl_divergence;
// reconstruction_loss = MSE(input, output)
// kl_divergence = -0.5 * sum(1 + log_var - mu^2 - exp(log_var))
```

**Components to Add**:
```
class VAE : extends Autoencoder
├── encode_distribution(shape)    // Returns (μ, σ)
├── reparameterize(mu, sigma)     // Sampling with gradient flow
├── train_vae(dataset, beta)      // Train with ELBO loss
└── sample_prior(n)               // Sample from N(0,I), decode
```

**Deliverables**:
- [ ] VAE class with reparameterization
- [ ] Latent space visualization (if 2D/3D latent)
- [ ] Sample quality comparison: AE vs VAE

---

### Phase 4: Kernel PCA (Nonlinear Classical) ⚠️ Priority: LOW

**Goal**: Implement nonlinear PCA without neural networks

**Components**:
```
src/kernel_pca.cpp + include/kernel_pca.hpp
├── class KernelPCA
│   ├── KernelPCA(kernel_type, kernel_params)
│   ├── fit(data_matrix)
│   ├── transform(shape)
│   ├── available kernels: RBF, polynomial, linear
│   └── kernel parameter tuning
```

**Challenge**: Pre-image problem (reconstruction is non-trivial)

**Deliverables**:
- [ ] Kernel matrix computation
- [ ] Kernel PCA projection
- [ ] Clustering/classification in kernel space

---

### Phase 5: LDDMM (Research/Advanced) 📚 Priority: EXPLORATORY

**Goal**: Implement diffeomorphic shape analysis

**Recommended Approach**: Python implementation using PyTorch + lagomorph

```python
# Python side (visualization/scripts/lddmm/)
├── atlas_building.py       # Build atlas from population
├── registration.py         # Register two shapes
├── geodesic_shooting.py    # Compute geodesics
└── pca_on_momenta.py       # Tangent space PCA
```

**C++ Integration** (if needed):
```
Option 1: pybind11 bindings to call C++ Femur class from Python
Option 2: File-based exchange (save/load OBJ files)
```

**Components**:
```
LDDMM Module (Python)
├── Kernel definition (Gaussian, Cauchy, etc.)
├── EPDiff integration (forward shooting)
├── Adjoint equations (backward gradient)
├── Matching functional optimization
└── Atlas construction loop
```

**Deliverables**:
- [ ] Basic registration between two femurs
- [ ] Atlas computation from population
- [ ] Visualization of deformation fields

---

## Directory Structure

```
Femur_Modeling/
├── include/
│   ├── femur.hpp          # ✅ Exists
│   ├── linalg.hpp         # ✅ Exists (Eigen-based)
│   ├── neuralNetwork.hpp  # ✅ Exists
│   ├── pca.hpp            # 🆕 To create
│   ├── autoencoder.hpp    # 🆕 To create
│   ├── dataset.hpp        # 🆕 To create (data loading utilities)
│   └── kernels.hpp        # 🆕 To create (kernel functions)
│
├── src/
│   ├── femur.cpp          # ✅ Exists
│   ├── linalg.cpp         # ✅ Exists
│   ├── neuralNetwork.cpp  # ✅ Exists
│   ├── pca.cpp            # 🆕 To create
│   ├── autoencoder.cpp    # 🆕 To create
│   ├── dataset.cpp        # 🆕 To create
│   └── main.cpp           # ✅ Exists (demo/entry point)
│
├── tests/
│   ├── testPCA.cpp            # 🆕 To create
│   ├── testAutoencoder.cpp    # 🆕 To create
│   └── ... (existing tests)
│
├── scripts/
│   ├── femur_standardization.py  # ✅ Exists
│   ├── train_autoencoder.py      # 🆕 For training control
│   └── lddmm/                    # 🆕 LDDMM Python module
│       ├── __init__.py
│       ├── atlas.py
│       └── registration.py
│
├── visualization/
│   ├── viewer3D.py        # ✅ Exists
│   ├── plot_modes.py      # 🆕 Visualize PCA modes
│   └── plot_latent.py     # 🆕 Visualize latent space
│
├── docs/
│   ├── SSA_Methods_Overview.md   # ✅ Just created
│   └── Architecture_Plan.md     # ✅ This file
│
├── data/
│   ├── training/          # ✅ Exists
│   └── validation/        # ✅ Exists
│
├── models/                # 🆕 Saved trained models
│   ├── pca_model.bin
│   └── autoencoder_model.bin
│
└── build/                 # ✅ Exists (CMake build)
```

---

## API Design

### Data Loading

```cpp
// Load all femurs from a directory
class FemurDataset {
public:
    FemurDataset(const std::string& directory);
    
    Matrix2D<double> getShapeMatrix() const;  // D x M matrix
    Femur getMean() const;
    size_t numSamples() const;
    size_t numVertices() const;
    
    // Convert between matrix column and Femur object
    static Vector<double> femurToVector(const Femur& f);
    static Femur vectorToFemur(const Vector<double>& v, const Femur& template_mesh);
};
```

### PCA

```cpp
template<typename T>
class PCA {
public:
    void fit(const Matrix2D<T>& data);  // data: D x M
    
    Vector<T> transform(const Vector<T>& shape, int n_components = -1) const;
    Vector<T> inverseTransform(const Vector<T>& code) const;
    
    Vector<T> getMean() const;
    Matrix2D<T> getComponents() const;  // D x K
    Vector<T> getExplainedVarianceRatio() const;
    
    // Generate new shapes
    Matrix2D<T> sample(int n_samples, int n_components) const;
    
    void save(const std::string& filename) const;
    void load(const std::string& filename);
};
```

### Autoencoder

```cpp
template<typename T>
class Autoencoder {
public:
    Autoencoder(const std::vector<size_t>& encoder_layers,
                size_t latent_dim,
                const std::string& activation = "relu");
    
    Vector<T> encode(const Vector<T>& input) const;
    Vector<T> decode(const Vector<T>& latent) const;
    Vector<T> reconstruct(const Vector<T>& input) const;
    
    void train(const Matrix2D<T>& data, 
               int epochs, 
               T learning_rate,
               int batch_size = 1);
    
    T getReconstructionError(const Vector<T>& input) const;
    
    // Interpolation in latent space
    Vector<T> interpolate(const Vector<T>& shape1, 
                          const Vector<T>& shape2, 
                          T alpha) const;
};
```

---

## Milestones & Timeline

### Milestone 1: PCA Implementation (Week 1-2)
- [ ] Implement `FemurDataset` class
- [ ] Implement `PCA` class with SVD
- [ ] Test on femur data
- [ ] Visualize deformation modes

### Milestone 2: Autoencoder (Week 3-4)
- [ ] Create `Autoencoder` class extending `NeuralNetwork`
- [ ] Train on femur data
- [ ] Compare with PCA (reconstruction error vs latent dimension)
- [ ] Implement shape interpolation

### Milestone 3: Advanced Features (Week 5-6)
- [ ] Implement VAE (optional)
- [ ] Partial shape reconstruction
- [ ] Constrained shape generation

### Milestone 4: LDDMM Exploration (Week 7-8)
- [ ] Python LDDMM prototype
- [ ] Atlas building
- [ ] Integration with visualization

### Milestone 5: Report & Documentation (Final Week)
- [ ] Complete technical report
- [ ] Code documentation
- [ ] Demo application

---

## Testing Strategy

### Unit Tests
- Matrix operations (existing)
- Neural network forward/backward (existing)
- PCA on synthetic data
- Autoencoder reconstruction on simple shapes

### Integration Tests
- Full pipeline: Load OBJ → PCA → Synthesize → Save OBJ
- Round-trip error measurement
- Mode visualization checks

### Benchmarks
- PCA computation time
- Neural network training time
- Reconstruction error vs latent dimension curves

---

## Dependencies

### Current
- **Eigen 5.0.0**: Linear algebra (already in `lib/`)
- **Standard C++17**: Core language features

### Recommended Additions
- **Python 3.x**: For visualization and LDDMM
- **PyTorch**: For LDDMM (Python-based)
- **matplotlib**: Visualization
- **pybind11** (optional): C++/Python bindings

---

## Key Decisions

| Decision | Recommendation | Rationale |
|----------|----------------|-----------|
| PCA implementation | Use Eigen's SVD | Already integrated, efficient |
| Autoencoder architecture | Start with 3-layer encoder | Small dataset, avoid overfitting |
| Activation function | ReLU for hidden, linear for output | Standard for autoencoders |
| LDDMM implementation | Python/PyTorch | Existing libraries (lagomorph), easier prototyping |
| Model serialization | Binary format | Already implemented for neural networks |

---

*Document created: January 2026*  
*Project: Femur Shape Modeling*
