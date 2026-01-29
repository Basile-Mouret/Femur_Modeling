# Medical Femur Analysis Tool

A command-line tool for analyzing a patient's femur shape compared to the population mean (atlas) using Tangent PCA, with **anatomically meaningful component interpretation**.

## Features

- **Fréchet (geodesic) distance** - Measures overall shape deviation using RKHS norm
- **Euclidean metrics** - L2 total, RMSE, and maximum per-vertex deviation
- **Anatomical component analysis** - Each PC mapped to clinically relevant shape features
- **Statistical severity levels** - Deviations classified based on population statistics
- **Clinical findings** - Automatic reporting of notable deformations with clinical context
- **Deviation heatmap** - 3D visualization of local deformations
- **Export options** - JSON report and PNG component chart

## Anatomical Components

The tool maps the first 7 principal components to anatomically meaningful shape variations:

| PC | Anatomy | Description | + Direction | - Direction | Significance |
|----|---------|-------------|-------------|-------------|--------------|
| PC1 | Size | Overall bone scale | Larger | Smaller | High |
| PC2 | Proportions (allometric) | Length vs thickness ratio | Shorter & thicker | Longer & thinner | High |
| PC3 | Torsion (mixed) | Anteversion + condylar width | More anteversion | Retroversion | High |
| PC4 | Extremities (mixed/noisy) | Extremity shape variation | — | — | Low (noisy) |
| PC5 | Cond. Tilt | Distal condyle angulation | Tilt variation | Tilt variation | Medium |
| PC6 | Neck (mixed) | Neck length + offset | Longer neck | Shorter neck | Medium |
| PC7 | Distal (subtle) | Subtle distal variations | — | — | Low (noise) |

**Note**: Components marked "mixed" capture correlated variations, not pure anatomical features. Components with "Low" significance are noisy and excluded from clinical findings unless extreme.

## Severity Levels

Deviations are classified based on standard normal distribution:

| Level | Range | Symbol | Population % |
|-------|-------|--------|--------------|
| Normal | <0.5σ | ✓ | 38% |
| Typical | 0.5-1σ | ○ | 30% |
| Mild | 1-1.5σ | △ | 18% |
| Moderate | 1.5-2σ | ▲ | 9% |
| Significant | 2-2.5σ | ◆ | 4% |
| Extreme | ≥2.5σ | ★ | 1% |

## Usage

```bash
# Basic analysis with visualization
python medical_femur_analysis.py <patient_femur.obj>

# Save report to directory
python medical_femur_analysis.py <patient_femur.obj> --output report/

# Quick analysis (no visualization)
python medical_femur_analysis.py <patient_femur.obj> --no-visualize

# Skip matplotlib chart
python medical_femur_analysis.py <patient_femur.obj> --no-chart
```

## Examples

```bash
# Analyze femur 13
python medical_femur_analysis.py data/training/L_Femur_13_DECIM.obj.FINAL.obj

# Analyze and save full report
python medical_femur_analysis.py data/training/L_Femur_11_DECIM.obj.FINAL.obj \
    --output report/femur_analysis/

# Quick check without opening windows
python medical_femur_analysis.py data/training/L_Femur_23_DECIM.obj.FINAL.obj \
    --no-visualize --no-chart
```

## Options

| Option | Description |
|--------|-------------|
| `femur` | Path to patient femur OBJ file (required) |
| `--model PATH` | Path to Tangent PCA model directory (default: `scripts/pca/model/tangent_pca`) |
| `--template PATH` | Template mesh for faces (default: `data/training/L_Femur_11_DECIM.obj.FINAL.obj`) |
| `--sigma FLOAT` | Kernel bandwidth for Fréchet distance (default: 10.0) |
| `--output DIR` | Output directory for JSON and PNG reports |
| `--no-visualize` | Skip 3D PyVista visualization |
| `--no-chart` | Skip matplotlib component chart |

## Output

### Console Report

```
======================================================================
FEMUR SHAPE ANALYSIS REPORT
======================================================================

Patient file: L_Femur_11_DECIM.obj.FINAL.obj
Mesh vertices: 18,291

--- Distance Metrics ---
  Fréchet (geodesic) distance: 133.11
  Euclidean L2 total:          1870.35
  Per-vertex RMSE:             13.83 mm
  Maximum deviation:           23.96 mm

  Overall assessment: SIGNIFICANT - substantial shape deviation

--- Anatomical Component Analysis ---
  PC     Component                σ        Level       
  -------------------------------------------------------
  PC1   Size                      -0.87  ○ Typical   
  PC2   Proportions (allometric)  +2.28  ◆ Significant
  PC3   Torsion (mixed)           +0.64  ○ Typical   
  PC4   Extremities (mixed/noisy) +1.82  ▲ Moderate  
  PC5   Cond. Tilt                +1.08  △ Mild      
  PC6   Neck (mixed)              -1.21  △ Mild      
  PC7   Distal (subtle)           +1.24  △ Mild      

--- Clinical Findings ---

  ◆ Significantly shorter and thicker bone proportions
     PC2: +2.28σ
     → Important for implant sizing, fracture risk assessment

  △ Mildly shorter femoral neck with reduced offset
     PC6: -1.21σ
     → Hip replacement offset selection, leg length, abductor mechanics

  △ Mildly atypical distal condyle angulation
     PC5: +1.08σ
     → Knee alignment, TKA component positioning

--- Deviation Level Legend ---
  ✓ Normal (<0.5σ)  ○ Typical (<1σ)  △ Mild (<1.5σ)
  ▲ Moderate (<2σ)  ◆ Significant (<2.5σ)  ★ Extreme (≥2.5σ)

======================================================================
```

**Note**: PC4 (Extremities) and PC7 (Distal) are excluded from clinical findings due to low component significance (noisy modes).

### JSON Output (`--output`)

```json
{
  "patient_file": "L_Femur_13_DECIM.obj.FINAL.obj",
  "n_vertices": 18291,
  "metrics": {
    "frechet_distance": 86.08,
    "l2_distance": 1230.68,
    "l2_rmse": 9.10,
    "l2_max": 15.53
  },
  "pca_components": {
    "PC1": {
      "coefficient_sigma": -0.94,
      "coefficient_raw": -1182.90,
      "variance_explained": 0.689
    },
    ...
  }
}
```

### PNG Chart (`--output`)

A matplotlib figure showing:
1. Horizontal bar chart with severity-colored bars and zone shading
2. Clinical findings panel with anatomical descriptions

### 3D Visualization

- Split-screen showing patient femur (left) and atlas/mean (right)
- Deviation heatmap on patient femur (blue=less, red=more deviation)
- Information panel with metrics and clinical findings
- Press 'q' to close

## Interpreting Results

### Overall RMSE Assessment

| RMSE | Assessment |
|------|------------|
| < 2 mm | NORMAL - within typical variation |
| 2-5 mm | MILD - slight deviation from mean |
| 5-10 mm | MODERATE - noticeable shape difference |
| > 10 mm | SIGNIFICANT - substantial shape deviation |

### Understanding the Analysis

1. **Anatomical Components**: Each PC represents a shape feature. Direction meanings vary by component (see table above).

2. **Clinical Findings**: Only components with ≥1σ deviation AND high/medium significance are reported. Low-significance (noisy) components are excluded unless extreme (≥2σ).

3. **Complete Sentences**: Findings are reported as readable sentences incorporating the direction and severity, e.g., "Significantly shorter and thicker bone proportions."

### Example Interpretations

- **PC2 +2.5σ (Proportions)**: "Significantly shorter and thicker" → may need shorter, wider implant
- **PC2 -1.8σ (Proportions)**: "Moderately longer and thinner" → may need longer, narrower implant  
- **PC3 -2.2σ (Torsion)**: "Significantly decreased anteversion (retroversion)" → rotational alignment in THA
- **PC6 -1.5σ (Neck)**: "Mildly shorter femoral neck" → offset considerations in hip replacement

## Requirements

- Tangent PCA model at `scripts/pca/model/tangent_pca/`
- Patient femur must have same vertex count and correspondence as training data
- Python packages: numpy, pyvista, trimesh, matplotlib, torch

## Related Tools

- **`tangent_pca_explorer.py`** - Interactively explore PCA components (press H for heatmap mode)
- **`frechet_distance.py`** - Compute distance between two arbitrary femurs
- **`pca_reconstruction_comparison.py`** - Compare Linear vs Tangent PCA reconstruction

## See Also

- [ANALYSIS_NOTES.md](ANALYSIS_NOTES.md) - Detailed PC analysis notes and femur rankings
