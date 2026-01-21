# LDDMM-based Tangent PCA for Femur Shape Analysis

This module provides tools for building statistical shape models using LDDMM (Large Deformation Diffeomorphic Metric Mapping) and Tangent PCA.

## Overview

The pipeline consists of:
1. **Data Loading** - Load femur meshes from OBJ files
2. **Atlas Building** - Compute the Fréchet mean (population average shape)
3. **Tangent PCA** - Perform PCA in the tangent space at the atlas
4. **Visualization** - Interactive exploration of shape variations

## Quick Start

### 1. Run the Full Pipeline

```bash
cd scripts/pca
source ../../.venv/bin/activate.fish  # or: source ../../.venv/bin/activate

python tangent_pca_demo.py
```

This will:
- Load all 22 femur shapes from `data/training/`
- Build the atlas (Fréchet mean)
- Fit Tangent PCA with 10 components
- Save the model to `scripts/pca/model/tangent_pca/`

### 2. Interactive Exploration

Launch the interactive explorer with slider controls:

```bash
python tangent_pca_explorer.py \
    --model model/tangent_pca \
    --template ../../data/training/L_Femur_11_DECIM.obj.FINAL.obj
```

**Controls:**
- **Sliders**: Adjust each principal component weight (±3σ)
- **R**: Reset all weights to zero (mean shape)
- **Q**: Quit
- **Mouse**: Rotate/zoom the 3D view

### 3. Visualization Commands

#### Show Atlas (Mean Shape)
```bash
python tangent_pca_visualizer.py \
    --model model/tangent_pca \
    --template ../../data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --atlas
```

#### Show Mode Variation (e.g., PC1)
```bash
python tangent_pca_visualizer.py \
    --model model/tangent_pca \
    --template ../../data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --mode 0
```

#### Show Multiple Modes in Grid
```bash
python tangent_pca_visualizer.py \
    --model model/tangent_pca \
    --template ../../data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --modes 5
```

#### Animate a Mode
```bash
python tangent_pca_visualizer.py \
    --model model/tangent_pca \
    --template ../../data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --animate 0
```

#### Variance Explained Plot
```bash
python tangent_pca_visualizer.py \
    --model model/tangent_pca \
    --template ../../data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --variance
```

#### Export Shapes to OBJ Files
```bash
python tangent_pca_visualizer.py \
    --model model/tangent_pca \
    --template ../../data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --export output/shapes/
```

#### Generate Full Report
```bash
python tangent_pca_visualizer.py \
    --model model/tangent_pca \
    --template ../../data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --report output/report/
```

## Module Structure

```
Femur_Modeling/
├── lddmm/                          # Core LDDMM module
│   ├── __init__.py                 # Package exports
│   ├── data_loader.py              # FemurDataLoader
│   ├── registration.py             # LDDMMPointRegistration
│   ├── atlas.py                    # LDDMMAtlasBuilder
│   ├── tangent_pca.py              # TangentPCA
│   ├── README.md                   # This file
│   └── tests/                      # LDDMM unit tests
│       ├── test_data_loader.py
│       ├── test_registration.py
│       ├── test_atlas.py
│       └── test_tangent_pca.py
├── scripts/
│   └── pca/                        # PCA visualization tools
│       ├── tangent_pca_visualizer.py   # TangentPCAVisualizer class
│       ├── tangent_pca_explorer.py     # Interactive explorer
│       ├── tangent_pca_demo.py         # Full pipeline demo
│       ├── test_tangent_pca_visualization.py
│       └── model/
│           └── tangent_pca/        # Saved model files
└── data/
    └── training/                   # Femur OBJ files
```

## Python API

### Loading a Saved Model

```python
from scripts.pca.tangent_pca_visualizer import load_tangent_pca_model, load_template_mesh, TangentPCAVisualizer

# Load model and template
model = load_tangent_pca_model('scripts/pca/model/tangent_pca')
template = load_template_mesh('data/training/L_Femur_11_DECIM.obj.FINAL.obj')

# Create visualizer
viz = TangentPCAVisualizer(model, template)

# Use it
viz.show_atlas()
viz.show_mode_variation(mode=0)
viz.plot_variance_explained()
```

### Building a New Model

```python
from lddmm import FemurDataLoader, LDDMMAtlasBuilder, TangentPCA

# Load data
loader = FemurDataLoader('data/training')
shapes, filenames = loader.load_all()

# Build atlas
builder = LDDMMAtlasBuilder(max_outer_iterations=5)
result = builder.build(shapes)
atlas = result['atlas']
momenta = result['momenta']

# Fit Tangent PCA
pca = TangentPCA(n_components=10)
pca.fit(atlas, momenta)

# Save
pca.save('scripts/pca/model/my_tangent_pca')
```

### Synthesizing New Shapes

```python
from lddmm import TangentPCA
import numpy as np

# Load model
pca = TangentPCA.load('scripts/pca/model/tangent_pca')

# Generate shape at specific PC weights
weights = np.array([2.0, -1.0, 0.5, 0, 0, 0, 0, 0, 0, 0])  # 10 components
shape = pca.synthesize_shape(weights)

# Generate shapes along a mode
shapes, t_values = pca.get_mode_extremes(mode=0, n_std=3, n_steps=5)
```

## Results

With 22 femur shapes (18,291 vertices each):

| Component | Variance | Cumulative |
|-----------|----------|------------|
| PC1       | 68.9%    | 68.9%      |
| PC2       | 16.4%    | 85.3%      |
| PC3       | 7.2%     | 92.5%      |
| PC4       | 2.1%     | 94.5%      |
| PC5       | 1.7%     | 96.2%      |

## Running Tests

```bash
source .venv/bin/activate.fish

# Run LDDMM tests
python -m pytest lddmm/tests/ -v

# Run visualization tests
python -m pytest scripts/pca/test_tangent_pca_visualization.py -v

# Run all tests (fast, excluding slow registration)
python -m pytest lddmm/tests/ scripts/pca/test_tangent_pca_visualization.py --ignore=lddmm/tests/test_registration.py -v
```

## Dependencies

- numpy
- scipy
- torch (with CUDA support recommended)
- pyvista
- trimesh
- matplotlib
- pytest (for testing)

## References

- Miller, Trouvé, Younes (2006) - Geodesic shooting for computational anatomy
- Fletcher et al. (2004) - Principal geodesic analysis on symmetric spaces
