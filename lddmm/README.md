# LDDMM Shape Analysis Module

Statistical shape analysis using Large Deformation Diffeomorphic Metric Mapping (LDDMM).

## Overview

This module provides a clean, well-documented implementation of:

1. **LDDMM Registration** - Geodesic shooting between shapes
2. **Atlas Building** - Fréchet mean computation
3. **Tangent PCA** - Principal component analysis on shape manifold

The implementation uses [scikit-shapes](https://scikit-shapes.github.io/scikit-shapes/) for true LDDMM geodesic shooting.

## Installation

```bash
pip install skshapes torch numpy trimesh
```

## Quick Start

```python
from lddmm import FemurDataLoader, AtlasBuilder, TangentPCA, LDDMMConfig

# 1. Load data
loader = FemurDataLoader("data/training")
shapes, filenames = loader.load_all()

# 2. Build atlas (mean shape) via geodesic averaging
config = LDDMMConfig.for_femurs()
builder = AtlasBuilder(config=config)
result = builder.build(shapes)
builder.save("model/atlas")

# 3. Fit Tangent PCA
pca = TangentPCA(n_components=10, config=config)
pca.fit(result.atlas, result.momenta)
pca.save("model/tangent_pca")

# 4. Synthesize new shapes
new_shapes = pca.synthesize_along_mode(mode=0, t_values=[-2, -1, 0, 1, 2])
```

## Theory

See [LDDMM_THEORY.md](LDDMM_THEORY.md) for comprehensive mathematical background.

### Key Concepts

**LDDMM (Large Deformation Diffeomorphic Metric Mapping)** computes smooth, invertible transformations between shapes by integrating a time-varying velocity field:

$$\frac{\partial \phi_t}{\partial t}(x) = v(\phi_t(x), t)$$

**Initial Momentum**: The velocity field is parametrized by an initial momentum $p_0$ at the source shape. This momentum is:
- The input to geodesic shooting (exponential map)
- The coordinate for tangent space PCA
- Lives in the cotangent space at the atlas

**Fréchet Mean**: The atlas $\mu$ minimizes total squared geodesic distance:

$$\mu = \arg\min_S \sum_{i=1}^K d^2(S, S_i)$$

For shapes with **point correspondence in Euclidean space**, this equals the arithmetic mean.

## API Reference

### Configuration

```python
from lddmm import LDDMMConfig

# Default configuration
config = LDDMMConfig()

# Presets
config = LDDMMConfig.for_femurs()     # Tuned for femur landmarks
config = LDDMMConfig.high_precision() # More accurate, slower
config = LDDMMConfig.fast()           # Quick exploratory analysis

# Custom
config = LDDMMConfig(
    n_steps=5,                  # Geodesic integration steps (≥5 for true LDDMM)
    kernel="gaussian",          # Kernel type: "gaussian" or "cauchy"
    scale=15.0,                 # Kernel bandwidth σ in mm
    regularization_weight=0.01, # Deformation smoothness penalty
    n_iter=100,                 # Optimizer iterations
    device="auto",              # "auto", "cuda", or "cpu"
)
```

### Data Loading

```python
from lddmm import FemurDataLoader, verify_correspondence, compute_bounding_box

# Load shapes
loader = FemurDataLoader("data/training", file_pattern="*.obj")
shapes, filenames = loader.load_all()

# Verify point correspondence
verify_correspondence(shapes)  # Returns True if all shapes have same vertex count

# Compute bounding box (useful for setting kernel scale)
bbox = compute_bounding_box(shapes)
print(f"Size: {bbox['size']}")  # e.g., [100, 80, 60] mm
```

### Registration

```python
from lddmm import LDDMMRegistration, LDDMMConfig

config = LDDMMConfig.for_femurs()
registration = LDDMMRegistration(config)

# Register source to target
result = registration.register(source, target)
print(result.momentum.shape)     # (N, 3) - initial momentum
print(result.transformed.shape)  # (N, 3) - deformed source
print(result.energy)             # Deformation energy

# Compute log map (registration shortcut)
momentum = registration.compute_momentum(source, target)

# Compute exponential map (geodesic shooting)
new_shape = registration.shoot(source, momentum)
```

### Atlas Building

```python
from lddmm import AtlasBuilder, LDDMMConfig

# Build Fréchet mean via iterative geodesic averaging
config = LDDMMConfig.for_femurs()
builder = AtlasBuilder(
    config=config,
    max_iterations=10,
)
result = builder.build(shapes)

# Access results
atlas = result.atlas       # (N, 3) mean shape
momenta = result.momenta   # (K, N, 3) momenta to each shape

# Save/load
builder.save("model/atlas")
loaded = AtlasBuilder.load("model/atlas")
```

### Tangent PCA

```python
from lddmm import TangentPCA

# Fit PCA
pca = TangentPCA(n_components=10)
pca.fit(atlas, momenta)

# Project shape to coefficients
coefficients = pca.project(shape)  # (n_components,)

# Synthesize shape from coefficients
new_shape = pca.synthesize_shape(coefficients)

# Synthesize along principal mode
shapes = pca.synthesize_along_mode(mode=0, t_values=[-2, -1, 0, 1, 2])

# Get mode extremes
shapes, t_values = pca.get_mode_extremes(mode=0, n_std=2.0)

# Reconstruct with fewer components
reconstructed = pca.reconstruct(shape, n_components=5)

# Explained variance
print(pca.explained_variance_ratio)  # [0.35, 0.20, 0.15, ...]
print(sum(pca.explained_variance_ratio[:5]))  # Cumulative for top 5

# Save/load
pca.save("model/tangent_pca")
loaded = TangentPCA.load("model/tangent_pca")
```

## Parameters

### Kernel Scale (σ)

The kernel scale controls the spatial correlation of deformations:

- **Small σ**: Local deformations, points move independently
- **Large σ**: Global deformations, nearby points move together

**Rule of thumb**: σ ≈ 10-20% of bounding box diagonal.

For femurs (~100mm bounding box): σ ≈ 10-20mm.

### Number of Steps (n_steps)

Controls the accuracy of geodesic integration:

- **n_steps=1**: Linear deformation (NOT true LDDMM)
- **n_steps≥5**: True LDDMM with geodesic shooting

Higher values give more accurate geodesics but slower computation.

### Regularization Weight

Controls the trade-off between matching accuracy and deformation smoothness:

- **High weight (0.1)**: Smoother deformations, may not match exactly
- **Low weight (0.001)**: Closer matching, may have irregular deformations

## File Structure

```
lddmm/
├── __init__.py          # Module exports
├── config.py            # LDDMMConfig dataclass
├── data_loader.py       # FemurDataLoader and utilities
├── registration.py      # LDDMMRegistration
├── atlas.py             # AtlasBuilder
├── tangent_pca.py       # TangentPCA
├── README.md            # This file
├── LDDMM_THEORY.md      # Mathematical background
├── IMPLEMENTATION_PLAN.md  # Migration plan
└── tests/
    ├── test_registration.py
    ├── test_atlas.py
    ├── test_tangent_pca.py
    └── test_data_loader.py
```

## Saved Model Files

### Atlas

```
model/atlas/
├── atlas.npy           # (N, 3) mean shape
├── momenta.npy         # (K, N, 3) initial momenta
└── atlas_metadata.json # Method, shape counts
```

### Tangent PCA

```
model/tangent_pca/
├── tangent_pca_atlas.npy              # (N, 3)
├── tangent_pca_mean_momentum.npy      # (N, 3)
├── tangent_pca_components.npy         # (n_components, N, 3)
├── tangent_pca_eigenvalues.npy        # (n_components,)
├── tangent_pca_explained_variance.npy # (n_components,)
└── tangent_pca_metadata.json
```

## True LDDMM Throughout

This implementation uses **true LDDMM geodesic operations** everywhere:

- **Atlas**: Computed via iterative geodesic averaging (Fréchet mean)
- **Log map**: LDDMM registration to compute initial momenta
- **Exponential map**: Geodesic shooting to synthesize shapes

See [LDDMM_THEORY.md](LDDMM_THEORY.md) for the mathematical foundations.

## References

1. Beg, M.F., et al. (2005). "Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms." *IJCV* 61(2), 139-157.

2. Younes, L. (2010). *Shapes and Diffeomorphisms*. Springer.

3. scikit-shapes: https://scikit-shapes.github.io/scikit-shapes/
