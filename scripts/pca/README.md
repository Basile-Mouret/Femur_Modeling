# 3D Femur Visualization & PCA Analysis

Tools for visualizing femur meshes and PCA-based Statistical Shape Models.

## Modules Overview

| Module | Description |
|--------|-------------|
| `viewer3D.py` | Basic OBJ file viewer |
| `pca_visualizer.py` | Comprehensive PCA visualization tools |
| `pca_explorer.py` | Interactive GUI with sliders for PCA exploration |
| `reconstruction_analysis.py` | Reconstruction quality analysis |
| `synthetic_data_generator.py` | Generate synthetic shapes from PCA model |

## Prerequisites

- **Python 3.10+** (tested with 3.12)
- A trained PCA model (`.bin` file from C++ implementation)

Check your Python version:
```bash
python3 --version
```

Install venv module if needed:
```bash
sudo apt update
sudo apt install python3-venv
```

## Installation

The project uses a single virtual environment at the repository root (`.venv`).

### 1. Activate Environment

```bash
# From repository root
cd /path/to/Femur_Modeling

# For fish shell
source .venv/bin/activate.fish

# For bash/zsh
source .venv/bin/activate
```

### 2. Install Dependencies (if needed)

```bash
pip install -r visualization/requirements.txt
```

## Usage

### Basic OBJ Viewer

```bash
python viewer3D.py
# Or use the test script
python test/testFemurViewer3D.py
```

### PCA Visualizations

All commands should be run from the **repository root** after activating `.venv`.

**Show mean shape:**
```bash
python visualization/pca_visualizer.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --mean
```

**Show mode of variation:**
```bash
python visualization/pca_visualizer.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --mode 0
```

**Show multiple modes in grid:**
```bash
python visualization/pca_visualizer.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --modes 5
```

**Animate a mode:**
```bash
python visualization/pca_visualizer.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --animate 0
```

**Show variance analysis plots:**
```bash
python visualization/pca_visualizer.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --variance
```

**Generate complete report:**
```bash
python visualization/pca_visualizer.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --report output/pca_report/
```

### Interactive Explorer

Launch the interactive PCA explorer with slider controls:

```bash
# From repository root
python visualization/pca_explorer.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --sliders 10 \
    --range 3.0
```

**Controls:**

| Control | Action |
|---------|--------|
| **Sliders** | Adjust principal component weights (in units of σ) |
| **Left-click + drag** | Rotate the 3D view |
| **Right-click + drag** | Pan the view |
| **Scroll wheel** | Zoom in/out |
| **R key** | Reset camera view |
| **Q key** | Quit the application |

**Understanding the Sliders:**

Each slider controls one principal component (PC). The slider value represents the number of standard deviations (σ) along that mode:

- **0**: Mean shape (no deformation)
- **+1 to +3**: Shape deformed in the positive direction of that mode
- **-1 to -3**: Shape deformed in the negative direction

The percentage shown next to each PC label indicates how much variance that component explains.

**Example Workflow:**

1. Start with all sliders at 0 (mean shape)
2. Move PC1 slider to +2σ to see the main mode of variation
3. Move PC1 back to 0, then adjust PC2 to explore the second mode
4. Combine multiple modes to create new shape variations

---

### Tangent PCA Explorer (LDDMM-based)

The Tangent PCA Explorer provides an advanced shape space exploration using LDDMM-based geodesic analysis. It includes a **heatmap mode** to visualize which regions of the femur are affected by each principal component.

```bash
# From repository root
python scripts/pca/tangent_pca_explorer.py \
    --model scripts/pca/model/tangent_pca \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --components 5 \
    --sigma 3.0
```

**Controls:**

| Control | Action |
|---------|--------|
| **Sliders** | Adjust principal component weights (in units of σ) |
| **H key** | **Toggle heatmap mode** (deviation from mean) |
| **R key** | Reset all sliders to mean shape |
| **Q key** | Quit the application |
| **Left-click + drag** | Rotate the 3D view |
| **Scroll wheel** | Zoom in/out |

**Heatmap Mode (Press H):**

When enabled, the mesh is colored by per-vertex deviation from the atlas (mean shape):

- **Blue regions** = minimal change from mean
- **Red regions** = high deviation from mean

This is extremely useful for **interpreting what each PC represents anatomically**:

1. Set all sliders to 0 (mean shape)
2. Press **H** to enable heatmap mode
3. Move a single slider (e.g., PC2) to ±2σ
4. Observe which regions turn red — those are the areas modified by that component

**Example: Interpreting PC2**

```bash
# Launch explorer
python scripts/pca/tangent_pca_explorer.py \
    --model scripts/pca/model/tangent_pca \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj
```

1. Press **H** to enable heatmap
2. Move PC2 slider to +2σ
3. Note which anatomical regions light up red (e.g., femoral head, greater trochanter, etc.)
4. Move PC2 to -2σ to see the opposite deformation
5. Press **R** to reset, then repeat for PC3, PC4, etc.

**Note:** PC1 typically represents overall scale/size variation. Focus on PC2 onwards for shape-specific deformations.

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--model`, `-m` | Path to Tangent PCA model directory | required |
| `--template`, `-t` | Path to template OBJ file | required |
| `--components`, `-c` | Number of PCs to control | 5 |
| `--sigma`, `-s` | Sigma range for sliders (±σ) | 3.0 |
| `--width` | Window width | 1400 |
| `--height` | Window height | 900 |

---

### Reconstruction Analysis

**Analyze a single shape:**
```bash
python visualization/reconstruction_analysis.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --shape data/validation/R_Femur_22_DECIM.obj.FINAL.obj
```

**Visualize reconstruction with error heatmap:**
```bash
python visualization/reconstruction_analysis.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --shape data/validation/R_Femur_22_DECIM.obj.FINAL.obj \
    --visualize --components 10
```

**Batch analyze all validation shapes:**
```bash
python visualization/reconstruction_analysis.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --batch data/validation/ \
    --output results/
```

### Synthetic Data Generator

Generate synthetic femur shapes from the trained PCA model for data augmentation.

**Generate random samples:**
```bash
python visualization/synthetic_data_generator.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --output data/synthetic \
    --count 100 \
    --strategy random \
    --seed 42
```

**Generate extreme mode variations (±2σ for each PC):**
```bash
python visualization/synthetic_data_generator.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --output data/synthetic/extremes \
    --strategy extreme \
    --sigma 2.0
```

**Latin Hypercube Sampling for comprehensive coverage:**
```bash
python visualization/synthetic_data_generator.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --output data/synthetic/lhs \
    --count 50 \
    --strategy lhs
```

**Grid sampling in first 3 PCs (5³ = 125 shapes):**
```bash
python visualization/synthetic_data_generator.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --output data/synthetic/grid \
    --strategy grid \
    --grid-dims 3 \
    --grid-points 5
```

**Generate all strategies at once:**
```bash
python visualization/synthetic_data_generator.py \
    --model visualization/model/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --output data/synthetic/all \
    --strategy all \
    --count 50 \
    --save-metadata
```

**Options:**

| Option | Description | Default |
|--------|-------------|---------|
| `--count` | Number of shapes (random/lhs) | 100 |
| `--strategy` | random, extreme, lhs, grid, all | random |
| `--components` | Number of PCs to use | all |
| `--sigma` | Sigma range for sampling | 3.0 |
| `--seed` | Random seed for reproducibility | None |
| `--grid-dims` | Dimensions for grid sampling | 3 |
| `--grid-points` | Points per dimension | 5 |
| `--save-metadata` | Export weights to JSON | False |

## Output Examples

### Variance Analysis Plot
Shows individual and cumulative variance explained by each principal component.

### Mode Variation Visualization
Displays shapes at -2σ, -1σ, 0, +1σ, +2σ along a principal component axis.

### Error Heatmap
3D visualization showing reconstruction error mapped onto the mesh surface.

## File Structure

```
visualization/
├── README.md                    # This file
├── requirements.txt             # Python dependencies
├── viewer3D.py                  # Basic OBJ viewer class
├── pca_visualizer.py            # PCA visualization module
├── pca_explorer.py              # Interactive PCA explorer
├── reconstruction_analysis.py   # Reconstruction quality analysis
├── synthetic_data_generator.py  # Synthetic shape generation
└── test/
    ├── testFemurViewer3D.py     # Test for basic viewer
    └── testPointsCloud.py       # Point cloud test
```

## API Reference

### PCAVisualizer Class

```python
from pca_visualizer import PCAVisualizer, load_pca_model, load_template_mesh

# Load model and template
model = load_pca_model('path/to/model.bin')
template = load_template_mesh('path/to/template.obj')

# Create visualizer
viz = PCAVisualizer(model, template)

# Visualize mean shape
viz.show_mean_shape()

# Visualize mode variation
viz.show_mode_variation(mode=0, sigma_range=(-2, 2), n_steps=5)

# Show multiple modes
viz.show_multiple_modes(n_modes=5, sigma=2.0)

# Animate a mode
viz.animate_mode(mode=0, sigma_range=(-3, 3), n_frames=60)

# Plot variance analysis
viz.plot_variance_explained()
viz.plot_mode_spectrum()

# Export shapes
viz.export_mean_shape('mean.obj')
viz.export_mode_variations('output/', n_modes=5)

# Generate complete report
viz.generate_report('report/')
```

### PCAExplorer Class

```python
from pca_explorer import PCAExplorer

explorer = PCAExplorer(
    model_path='path/to/model.bin',
    template_path='path/to/template.obj',
    n_sliders=10,
    sigma_range=3.0
)
explorer.run()
```

### ReconstructionAnalyzer Class

```python
from reconstruction_analysis import ReconstructionAnalyzer

analyzer = ReconstructionAnalyzer('model.bin', 'template.obj')

# Analyze single shape
results = analyzer.analyze_shape('shape.obj')

# Plot error curves
analyzer.plot_error_by_components('shape.obj')

# Visualize with error heatmap
analyzer.visualize_reconstruction('shape.obj', n_components=10)

# Batch analyze directory
all_results = analyzer.batch_analyze('data/', 'output/')
```

### SyntheticGenerator Class

```python
from synthetic_data_generator import SyntheticGenerator, load_pca_model

# Load model
model = load_pca_model('model.bin')

# Create generator
generator = SyntheticGenerator(
    model=model,
    template_path='template.obj',
    n_components=10,      # Use first 10 PCs (optional)
    sigma_range=3.0,      # ±3σ sampling range
    seed=42               # For reproducibility
)

# Generate samples with different strategies
random_samples = generator.generate_random(count=100)
extreme_samples = generator.generate_extreme_modes(sigma=2.0)
lhs_samples = generator.generate_lhs(count=50)
grid_samples = generator.generate_grid(n_dims=3, points_per_dim=5)

# Save to OBJ files
files = generator.save_shapes(random_samples, 'output/', prefix='synth')

# Get shape points directly
points = generator.weights_to_shape(weights=np.zeros(10))
```

## Tips

1. **Performance**: For large meshes, the first render may take a few seconds
2. **Display issues**: If running over SSH, use X forwarding (`ssh -X`)
3. **Headless mode**: Use `screenshot` parameter instead of `show()` for batch processing
4. **Memory**: Each shape uses ~1.3MB (54,873 vertices × 8 bytes × 3 coords)

## Troubleshooting

**"Display not found" error:**
```bash
export DISPLAY=:0
# Or use virtual framebuffer
Xvfb :99 &
export DISPLAY=:99
```

**PyVista rendering issues:**
```python
import pyvista
pyvista.start_xvfb()  # For headless rendering
```

**Visu with sliders**

```bash
source ../visualization/venv/bin/activate
cd build && cmake .. -Dpybind11_DIR=$(python3 -c "import pybind11; print(pybind11.get_cmake_dir())")
&& make femur_rdn
```



**For view_latent_3d**

```bash
./view_latent_3d.py -n -1
```