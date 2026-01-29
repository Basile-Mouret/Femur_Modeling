# Femur Shape Analysis Takeaways

## Dataset Overview
- **Total femurs analyzed**: 9 (training set)
- **PCA model**: Tangent PCA (LDDMM-based)
- **Components**: 10 principal components retained

---

## Most Deformed Femurs (excluding PC1/size)

| Rank | Femur | Total Deviation | Dominant Component |
|------|-------|-----------------|-------------------|
| 1 | **L_Femur_11** | 3.88 | PC2 (+2.28σ) |
| 2 | **L_Femur_15** | 3.20 | PC7 (+2.25σ) |
| 3 | **L_Femur_16** | 2.99 | PC5 (-1.94σ) |
| 4 | **L_Femur_23** | 2.75 | PC3 (-2.22σ) |
| 5 | L_Femur_14 | 2.65 | PC10 (+1.80σ) |
| 6 | L_Femur_20 | 2.55 | PC3 (-1.87σ) |
| 7 | L_Femur_13 | 2.46 | PC8 (+2.06σ) |
| 8 | L_Femur_21 | 2.37 | PC6 (+1.44σ) |
| 9 | L_Femur_19 | 2.31 | PC6 (+1.58σ) |

---

## Top Deformed Femurs by Individual Component (PC2-PC10)

| Component | Most Extreme Femur | Deviation | Direction |
|-----------|-------------------|-----------|-----------|
| **PC2** | L_Femur_11 | +2.28σ | positive |
| **PC3** | L_Femur_23 | -2.22σ | negative |
| **PC4** | L_Femur_11 | +1.82σ | positive |
| **PC5** | L_Femur_16 | -1.94σ | negative |
| **PC6** | L_Femur_19 | +1.58σ | positive |
| **PC7** | L_Femur_15 | +2.25σ | positive |
| **PC8** | L_Femur_13 | +2.06σ | positive |
| **PC9** | L_Femur_14 | +1.34σ | positive |
| **PC10** | L_Femur_14 | +1.80σ | positive |

---

## Recommended Examples for Presentation

### Most Deformed Overall
- **L_Femur_11** - Most deformed overall (3.88 total deviation), strong PC2 (+2.28σ) and PC4 (+1.82σ)

### By Component Extremes

| Component | Positive Extreme | Negative Extreme |
|-----------|-----------------|------------------|
| PC2 | L_Femur_11 (+2.28σ) | L_Femur_20 (-0.62σ) |
| PC3 | L_Femur_11 (+0.64σ) | L_Femur_23 (-2.22σ) |
| PC4 | L_Femur_11 (+1.82σ) | L_Femur_19 (-0.93σ) |

---

## Component Interpretation Notes

> **Method**: Components analyzed using Tangent PCA Explorer with heatmap mode (press H). Move individual PC sliders to ±3σ to observe deformation patterns.

| Component | Variance | Interpretation | + Direction | - Direction | Significance |
|-----------|----------|----------------|-------------|-------------|--------------|
| PC1 | 68.9% | **Overall Size** | Larger bone | Smaller bone | High |
| PC2 | 16.4% | **Proportions** (allometric) | Shorter & thicker | Longer & thinner | High |
| PC3 | 7.2% | **Torsion** (mixed) | More anteversion, wider condyles | Retroversion, narrower condyles | High |
| PC4 | 2.1% | **Extremity Shape** (mixed/noisy) | — | — | Low (noisy) |
| PC5 | 1.7% | **Condyle Tilt** | Tilt variation | Tilt variation | Medium |
| PC6 | 0.8% | **Neck Length** (mixed) | Longer neck, more offset | Shorter neck, less offset | Medium |
| PC7 | 0.7% | **Distal Detail** (subtle) | — | — | Low (noise) |

**Cumulative variance (PC1-PC5)**: ~96.3%

### Key Corrections (Jan 21, 2026)

- **PC2**: Direction clarified - positive = shorter & thicker (not longer)
- **PC4**: Renamed from "Shaft Bowing" to "Extremity Shape" - bowing not clearly visible, mostly noisy mixed variation
- **PC5**: Renamed from "Proximal Geometry" to "Condyle Tilt" - main variation is distal condyle angulation, not proximal
- **PC7**: Marked as likely noise at 0.7% variance

---

## Detailed Component Analysis

### PC1 (68.9% variance) - Overall Size/Scale

Global scaling of the femur. Likely correlates directly with patient height.

---

### PC2 (16.4% variance) - Allometric Scaling / Length-Thickness Ratio

**Observation**: Bone length varies while preserving the length-to-thickness ratio in a biomechanically meaningful way.

**Heatmap pattern**:
- 🔴 High deviation: Femoral head/neck, greater trochanter (proximal), condyles (distal)
- 🔵 Low deviation: Mid-shaft remains relatively stable

**Medical interpretation**:
- **Allometric scaling**: The bone grows proportionally, not uniformly stretched
- **Length-to-width ratio preservation**: Critical for load-bearing capacity (a longer femur that's too thin would buckle under stress)
- **Differential growth**: Proximal and distal regions change more than mid-shaft, consistent with epiphyseal plate growth patterns

**Clinical relevance**:
- May correlate with **body proportions** independent of overall height
- Important for **implant sizing** in hip/knee replacements
- Related to **cortical thickness ratio** affecting fracture risk

---

### PC3 (7.2% variance) - Femoral Torsion + Condylar Width

**Observation**: Combined rotational and width variation affecting both proximal and distal ends.

**Heatmap pattern**:
- 🔴 High deviation: Femoral condyles (distal), femoral head/neck region (proximal)
- 🔵 Low deviation: Mid-shaft relatively stable

**Changes observed**:
- **At -3σ**: Condyles appear narrower/more compressed; femoral neck more vertical
- **At +3σ**: Condyles wider/more spread apart; femoral neck more angled

**Medical interpretation - Femoral Anteversion/Torsion**:
- **Femoral anteversion**: The angle of the femoral neck relative to the transcondylar axis
  - Normal range: 10-15° in adults
  - Increased anteversion → internal rotation ("pigeon-toed" gait)
  - Decreased anteversion (retroversion) → external rotation ("duck-footed" gait)
- **Femoral torsion**: The cumulative twist along the shaft between proximal and distal ends

**Clinical relevance**:
- Important for **total hip arthroplasty** planning (stem anteversion)
- Related to **patellofemoral problems** and knee alignment
- Affects **gait mechanics** and lower limb rotation

**⚠️ Limitation**: This is a *mixed mode* - not purely anteversion or purely condylar width, but a statistically correlated combination. PCA captures covariance patterns, not anatomically "pure" deformations.

---

### PC4 (2.1% variance) - Extremity Shape (Mixed/Noisy)

**Observation**: Mixed variation in proximal and distal extremity proportions. Does not show clear anatomical pattern.

**Changes observed**:
- **At ±3σ**: Changes at proximal and distal ends, no consistent shaft bowing visible
- Pattern appears noisy without clear anatomical interpretation

**Note**: Originally thought to represent shaft bowing, but visual inspection shows this is a mixed/noisy mode.

**Clinical relevance**:
- May affect implant fit at bone extremities
- Low significance for clinical interpretation due to noise

**⚠️ Limitation**: This component appears to capture statistical noise rather than a pure anatomical feature. Excluded from clinical findings unless extreme (≥2σ).

---

### PC5 (1.7% variance) - Condyle Tilt

**Observation**: Angulation of the distal condyles (knee articulation surface). Main variation is in distal geometry, not proximal.

**Changes observed**:
- **At ±3σ**: Distal condyle angulation changes, affecting knee joint surface orientation
- Primarily affects the medial/lateral balance of the condyles

**Medical interpretation**:
- **Condyle angulation**: The tilt of the distal femoral condyles relative to the shaft axis
- Affects knee joint congruence with tibial plateau
- Related to varus/valgus knee alignment

**Clinical relevance**:
- **Total knee arthroplasty**: Femoral component rotation and alignment
- **Knee alignment**: Assessment of mechanical axis
- **Patellofemoral tracking**: Trochlear groove orientation

**⚠️ Limitation**: Medium significance - captures real anatomical variation but at low variance.

---

### PC6 (0.8% variance) - Neck Length + Distal Geometry

**Observation**: Changes in the distance from greater trochanter attachment to the femoral head, combined with subtle distal condyle shape variation.

**Changes observed**:
- **At -3σ**: Shorter apparent neck length, altered condyle morphology
- **At +3σ**: Longer neck region, different distal shape

**Medical interpretation**:
- **Femoral offset**: Perpendicular distance from femoral shaft axis to center of femoral head
  - Affects hip abductor mechanics and joint reaction forces
- **Neck length**: Important for range of motion and stability

**Clinical relevance**:
- **Hip replacement offset selection**: Critical for soft tissue tension and gait
- **Leg length discrepancy**: Neck length affects overall limb length
- Affects **hip joint biomechanics** and muscle lever arms

**⚠️ Limitation**: Subtle variations coupling proximal and distal changes - likely captures residual covariance.

---

### PC7 (0.7% variance) - Distal Condyle Morphology

**Observation**: Subtle variations primarily affecting the distal femur and condyle region.

**Changes observed**:
- Shape changes in the medial and lateral condyles
- Alterations in the intercondylar notch region
- Minor thickness variations in distal shaft

**Medical interpretation**:
- **Condylar geometry**: Affects knee joint congruence with tibial plateau
- **Trochlear groove**: Patellofemoral tracking surface
- **Intercondylar notch**: Space for cruciate ligaments

**Clinical relevance**:
- **Total knee arthroplasty sizing**: Femoral component selection
- **ACL reconstruction**: Notch width affects graft placement
- **Patellofemoral disorders**: Trochlear dysplasia assessment

**⚠️ Limitation**: Small variance (0.7%) means this captures subtle population variation, potentially approaching noise levels. Anatomical significance should be interpreted cautiously.

---

## Analysis Tools

- `medical_femur_analysis.py` - Analyze individual femur vs atlas
- `pca_reconstruction_comparison.py` - Compare Linear vs Tangent PCA reconstruction
- `frechet_distance.py` - Compute geodesic distance between two shapes
- `tangent_pca_explorer.py` - Interactive visualization of PCA modes

---

*Last updated: January 21, 2026*
