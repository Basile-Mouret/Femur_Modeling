# 3D Femur Visualization & PCA Analysis

Tools for visualizing femur meshes and PCA-based Statistical Shape Models.

## Modules Overview

| Module | Description |
|--------|-------------|
| `viewer3D.py` | Basic OBJ file viewer |
| `pca_visualizer.py` | Comprehensive PCA visualization tools |
| `pca_explorer.py` | Interactive GUI with sliders for PCA exploration |
| `reconstruction_analysis.py` | Reconstruction quality analysis |

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
    --model bin/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --mean
```

**Show mode of variation:**
```bash
python visualization/pca_visualizer.py \
    --model bin/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --mode 0
```

**Show multiple modes in grid:**
```bash
python visualization/pca_visualizer.py \
    --model bin/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --modes 5
```

**Animate a mode:**
```bash
python visualization/pca_visualizer.py \
    --model bin/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --animate 0
```

**Show variance analysis plots:**
```bash
python visualization/pca_visualizer.py \
    --model bin/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --variance
```

**Generate complete report:**
```bash
python visualization/pca_visualizer.py \
    --model bin/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --report output/pca_report/
```

### Interactive Explorer

Launch the interactive PCA explorer with slider controls:

```bash
# From repository root
python visualization/pca_explorer.py \
    --model bin/pca_femur_model.bin \
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

### Reconstruction Analysis

**Analyze a single shape:**
```bash
python visualization/reconstruction_analysis.py \
    --model bin/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --shape data/validation/R_Femur_22_DECIM.obj.FINAL.obj
```

**Visualize reconstruction with error heatmap:**
```bash
python visualization/reconstruction_analysis.py \
    --model bin/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --shape data/validation/R_Femur_22_DECIM.obj.FINAL.obj \
    --visualize --components 10
```

**Batch analyze all validation shapes:**
```bash
python visualization/reconstruction_analysis.py \
    --model bin/pca_femur_model.bin \
    --template data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --batch data/validation/ \
    --output results/
```

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
