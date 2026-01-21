# LDDMM Algorithm Documentation
## Large Deformation Diffeomorphic Metric Mapping for Statistical Shape Analysis

**Author:** Femur Modeling Project  
**Date:** 2024  
**Purpose:** Comprehensive documentation of LDDMM algorithm with mathematical foundations, critical analysis, and comparison with linear methods (PCA)

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [Mathematical Foundations](#2-mathematical-foundations)
   - [2.1 Diffeomorphisms](#21-diffeomorphisms)
   - [2.2 Riemannian Geometry on Shape Spaces](#22-riemannian-geometry-on-shape-spaces)
   - [2.3 Reproducing Kernel Hilbert Spaces (RKHS)](#23-reproducing-kernel-hilbert-spaces-rkhs)
3. [The LDDMM Algorithm](#3-the-lddmm-algorithm)
   - [3.1 Flow Equation](#31-flow-equation)
   - [3.2 Energy Functional](#32-energy-functional)
   - [3.3 Geodesic Shooting](#33-geodesic-shooting)
   - [3.4 EPDiff Equation](#34-epdiff-equation)
4. [Numerical Implementation](#4-numerical-implementation)
   - [4.1 Discretization](#41-discretization)
   - [4.2 Gradient Descent Optimization](#42-gradient-descent-optimization)
   - [4.3 Flow Integration](#43-flow-integration)
5. [LDDMM for Point Clouds and Surfaces](#5-lddmm-for-point-clouds-and-surfaces)
6. [Statistical Shape Analysis with LDDMM](#6-statistical-shape-analysis-with-lddmm)
   - [6.1 Atlas Building](#61-atlas-building)
   - [6.2 Tangent Space Statistics](#62-tangent-space-statistics)
   - [6.3 PCA in Deformation Space](#63-pca-in-deformation-space)
7. [Comparison: LDDMM vs Linear PCA](#7-comparison-lddmm-vs-linear-pca)
8. [Critical Analysis](#8-critical-analysis)
9. [Applications and Capabilities](#9-applications-and-capabilities)
10. [References](#10-references)

---

## 1. Introduction

**Large Deformation Diffeomorphic Metric Mapping (LDDMM)** is a mathematical framework for computing diffeomorphic transformations between shapes while preserving their topology. It was introduced by Beg et al. (2005) as an extension of viscous fluid models to provide:

- **Diffeomorphic transformations**: Smooth, invertible mappings that preserve topology
- **Riemannian metric**: A proper distance on the space of shapes
- **Geodesic paths**: Optimal deformation trajectories between shapes
- **Statistical framework**: Proper mathematical structure for shape statistics

Unlike linear methods like PCA, LDDMM operates on the nonlinear manifold of shapes, capturing complex anatomical variations while guaranteeing biologically meaningful transformations.

### Why LDDMM for Femur Modeling?

For anatomical structures like the femur:
- **Topology preservation**: Bone surfaces must remain closed manifolds without self-intersections
- **Large deformations**: Inter-subject variability can be substantial
- **Physical plausibility**: Deformations should represent realistic anatomical variations
- **Statistical analysis**: Population studies require proper mathematical foundations

---

## 2. Mathematical Foundations

### 2.1 Diffeomorphisms

A **diffeomorphism** $\phi: \Omega \to \Omega$ is a smooth bijective map with a smooth inverse. The group of diffeomorphisms $\text{Diff}(\Omega)$ forms an infinite-dimensional Lie group.

**Key Properties:**
- **Invertibility**: $\phi^{-1}$ exists and is smooth
- **Composition**: $\phi_1 \circ \phi_2 \in \text{Diff}(\Omega)$
- **Topology preservation**: Connected components, holes, and boundaries are preserved

**Jacobian Constraint:**
For a diffeomorphism, the Jacobian determinant must be strictly positive everywhere:
$$\det(D\phi(x)) > 0 \quad \forall x \in \Omega$$

This ensures local injectivity (no folding of the transformation).

### 2.2 Riemannian Geometry on Shape Spaces

The space of shapes $\mathcal{M}$ is endowed with a Riemannian structure through the action of diffeomorphisms.

**The Lie Algebra:**
The Lie algebra of $\text{Diff}(\Omega)$ consists of smooth vector fields $V = \{v: \Omega \to \mathbb{R}^d\}$.

**Inner Product (Metric):**
We define an inner product on $V$ using a differential operator $L$:
$$\langle v, w \rangle_V = \int_\Omega (Lv)(x) \cdot (Lw)(x) \, dx = \langle Lv, Lw \rangle_{L^2}$$

**Common Operator (Cauchy-Navier / Biharmonic):**
$$L = (-\alpha \Delta + \gamma I)^p$$

Where:
- $\alpha > 0$: Smoothness parameter (higher = smoother deformations)
- $\gamma > 0$: Norm penalty ensuring non-singularity
- $p \geq 1$: Power (typically $p=2$ for biharmonic)
- $\Delta$: Laplacian operator

**The Kernel Operator:**
The kernel $K = L^{-1}$ smooths the vector field:
$$v = K \cdot m \quad \text{where } m = Lv$$

Here $m$ is the **momentum** (dual variable to velocity).

### 2.3 Reproducing Kernel Hilbert Spaces (RKHS)

The space $V$ equipped with $\langle \cdot, \cdot \rangle_V$ forms a **Reproducing Kernel Hilbert Space (RKHS)**.

**Reproducing Property:**
$$v(x) = \langle v, K_x \rangle_V$$

**Gaussian Kernel (common choice):**
$$K(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right) I_d$$

**Physical Interpretation:**
- The kernel controls the spatial correlation of the velocity field
- Points closer than $\sigma$ (kernel width) move together
- Larger $\sigma$ produces smoother, more global deformations

---

## 3. The LDDMM Algorithm

### 3.1 Flow Equation

A diffeomorphism is generated by integrating a time-dependent velocity field $v_t$ over $t \in [0,1]$:

$$\frac{d\phi_t}{dt}(x) = v_t(\phi_t(x)), \quad \phi_0 = \text{Id}$$

The final transformation $\phi_1$ maps the source shape to the target.

**Inverse Flow:**
The inverse $\phi_t^{-1}$ (backward flow) satisfies:
$$\frac{d\phi_t^{-1}}{dt}(x) = -D\phi_t^{-1}(x) \cdot v_t(x)$$

### 3.2 Energy Functional

LDDMM minimizes a **variational energy** balancing regularity and matching:

$$E(v) = \underbrace{\frac{1}{2\sigma_R^2} \int_0^1 \|v_t\|_V^2 \, dt}_{\text{Regularity}} + \underbrace{\frac{1}{2\sigma_M^2} A(\phi_1 \cdot S, T)}_{\text{Matching}}$$

**Regularity Term:**
$$\|v_t\|_V^2 = \langle Lv_t, Lv_t \rangle_{L^2}$$

This is the **kinetic energy** of the flow, measuring the "cost" of the deformation.

**Matching Term Examples:**
- **Image matching (L²)**: $A(\phi \cdot I, J) = \int_\Omega |I(\phi^{-1}(x)) - J(x)|^2 \, dx$
- **Point cloud matching**: $A = \sum_i \|T_i - \phi(S_i)\|^2$
- **Current/Varifold matching**: For surfaces without correspondence

**Parameters:**
- $\sigma_R$: Regularization weight (higher = smoother paths)
- $\sigma_M$: Matching tolerance (higher = more flexibility in matching)

### 3.3 Geodesic Shooting

A fundamental result: **minimizing geodesics are characterized by their initial conditions**.

The geodesic equation shows that if $v_t^*$ is optimal, then the **momentum** $m_t = Lv_t$ is transported by the flow:

**Conservation Law (EPDiff):**
$$\frac{\partial m_t}{\partial t} + \text{ad}^*_{v_t} m_t = 0$$

This means: **knowing $m_0$ determines the entire geodesic path**.

**Practical Consequence:**
- Shapes are parameterized by initial momenta $m_0$
- Dimension reduction: from infinite-dimensional $v_t$ to $m_0$
- Enables tangent space statistics

### 3.4 EPDiff Equation

The **Euler-Poincaré equation for diffeomorphisms (EPDiff)** describes geodesic evolution:

$$\frac{\partial m}{\partial t} = -\nabla v^T m - \nabla \cdot (m v^T) - (m \cdot \nabla) v$$

In component form (3D):
$$\frac{\partial m_i}{\partial t} = -\sum_j \left( v_j \frac{\partial m_i}{\partial x_j} + m_j \frac{\partial v_j}{\partial x_i} + m_i \frac{\partial v_j}{\partial x_j} \right)$$

**Key Properties:**
- **Momentum conservation**: $\int m_t \, dx$ is conserved
- **Energy conservation**: $\|v_t\|_V^2$ is constant along geodesics
- **Hamiltonian structure**: The system is symplectic

---

## 4. Numerical Implementation

### 4.1 Discretization

**Time Discretization:**
Divide $[0,1]$ into $T$ timesteps: $t_k = k/T$, $k = 0, \ldots, T-1$

**Spatial Discretization:**
- **Images**: Regular grid of $H \times W \times D$ voxels
- **Point clouds**: $N$ points $\{x_i\}_{i=1}^N$
- **Velocity field**: Typically on a coarser grid than the image

**FFT-Based Operators:**
The regularization operator $L$ and its inverse $K$ are efficiently computed in Fourier space:

$$\hat{L}(\omega) = \left(\gamma + \alpha \|\omega\|^2\right)^p$$

$$v = K * m = \mathcal{F}^{-1}\left(\frac{\hat{m}}{\hat{L}}\right)$$

### 4.2 Gradient Descent Optimization

**Algorithm (Beg et al., 2005):**

```
Input: Source I₀, Target I₁, parameters σ, α, γ, ε, T, K_max
Initialize: v = 0 (velocity field)

for k = 1 to K_max:
    1. Reparameterize v (optional, every ~10 iterations)
    
    2. Compute backward flow Φ₁:
       Φ₁[T-1] = Id
       for t = T-2 to 0:
           Φ₁[t] = Φ₁[t+1](x + α(v[t], x))
    
    3. Compute forward flow Φ₀:
       Φ₀[0] = Id
       for t = 0 to T-2:
           Φ₀[t+1] = Φ₀[t](x - α(v[t], x))
    
    4. Push-forward source: J₀ = I₀ ∘ Φ₀⁻¹
    
    5. Pull-back target: J₁ = I₁ ∘ Φ₁
    
    6. Compute image gradients: ∇J₀
    
    7. Compute Jacobian determinant: det(DΦ₁)
       Check injectivity: if min(det(DΦ₁)) < 0, stop
    
    8. Compute gradient:
       ∂E/∂v[t] = 2v[t] - K * (2/σ² · det(DΦ₁) · ∇J₀ · (J₀ - J₁))
    
    9. Update: v ← v - ε · ∂E/∂v
    
    10. Compute energy and check convergence

Output: Final velocity field v̂, transformation Φ
```

### 4.3 Flow Integration

**Semi-Lagrangian Method:**
To compute $\phi_{t+dt}$ from $\phi_t$:

$$\phi_{t+dt}(x) = \phi_t(x - v_t(x) \cdot dt)$$

**RK4 Integration (more accurate):**
For better stability with larger timesteps.

**Scaling and Squaring:**
For stationary velocity fields (SVF), compute $\exp(v)$ efficiently:
$$\phi = \exp(v) = \exp(v/2^n)^{2^n}$$

---

## 5. LDDMM for Point Clouds and Surfaces

For our femur data (point clouds with $N = 18,291$ vertices), specialized formulations exist:

### Current Matching

Surfaces are represented as **currents** (geometric measure theory):
$$S = \sum_i \delta_{c_i} \otimes n_i$$

Where $c_i$ is the triangle centroid and $n_i$ is the weighted normal.

**Kernel Matching:**
$$A(S, T) = \|S - T\|_{W^*}^2 = \langle S, K_W * S \rangle - 2\langle S, K_W * T \rangle + \langle T, K_W * T \rangle$$

**Advantage:** No point correspondence required.

### Varifold Matching

More robust than currents for surfaces with orientation ambiguity:
$$A = \int \int k_p(p, q) k_s(\vec{n}_p, \vec{n}_q) \, d\mu_S(p) d\mu_T(q)$$

### Point Cloud with Correspondence

If correspondence is known (our case: all femurs have 18,291 aligned vertices):
$$A(\phi, S, T) = \sum_{i=1}^N \|\phi(S_i) - T_i\|^2$$

This is simpler and computationally cheaper than current/varifold matching.

---

## 6. Statistical Shape Analysis with LDDMM

### 6.1 Atlas Building

**Goal:** Compute a reference shape (atlas) $\bar{S}$ that minimizes total deformation to all shapes $\{S_j\}_{j=1}^N$.

**Objective:**
$$\bar{S} = \arg\min_S \sum_{j=1}^N d_G(S, S_j)^2$$

Where $d_G$ is the geodesic distance.

**Iterative Algorithm:**
```
1. Initialize atlas S̄ (e.g., one of the shapes or their average)
2. Repeat until convergence:
   a. For each shape Sⱼ:
      Compute geodesic shooting from S̄ to Sⱼ
      Get initial momentum m₀ʲ
   b. Compute mean momentum: m̄ = (1/N) Σⱼ m₀ʲ
   c. Update atlas: S̄ ← shoot from S̄ with momentum m̄
```

**Result:**
- Atlas $\bar{S}$: Population mean shape (Fréchet mean)
- Initial momenta $\{m_0^j\}$: Encoding of each shape's deviation from mean

### 6.2 Tangent Space Statistics

Once the atlas is built, each shape $S_j$ is represented by its initial momentum $m_0^j$:

$$S_j \approx \exp_{\bar{S}}(m_0^j)$$

The collection $\{m_0^j\}$ lives in the **tangent space** $T_{\bar{S}}\mathcal{M}$, which is a **vector space**.

**This enables:**
- PCA in tangent space (proper nonlinear generalization)
- Linear statistics on inherently nonlinear data
- Extrapolation along principal geodesics

### 6.3 PCA in Deformation Space

**Tangent PCA:**
1. Center momenta: $\tilde{m}^j = m_0^j - \bar{m}$
2. Compute covariance: $C = \frac{1}{N-1} \sum_j \tilde{m}^j \otimes \tilde{m}^j$
3. Eigendecomposition: $C \xi_k = \lambda_k \xi_k$
4. Principal geodesics: $\gamma_k(t) = \exp_{\bar{S}}(t \cdot \xi_k)$

**Interpretation:**
- $\xi_k$: Principal modes of deformation
- $\lambda_k$: Variance along mode $k$
- $\gamma_k(t)$: Shape variations along principal direction

---

## 7. Comparison: LDDMM vs Linear PCA

### Summary Table

| Aspect | Linear PCA | LDDMM |
|--------|------------|-------|
| **Geometry** | Euclidean (flat) | Riemannian (curved) |
| **Shape Space** | Vector space $\mathbb{R}^{3N}$ | Manifold $\text{Diff}(\Omega)$ |
| **Transformations** | Implicit (linear interpolation) | Explicit diffeomorphisms |
| **Topology** | Not guaranteed | Preserved by construction |
| **Distance** | $L^2$ norm | Geodesic distance |
| **Mean Shape** | Arithmetic average | Fréchet/Karcher mean |
| **Interpolation** | Linear (may cause artifacts) | Geodesic (smooth, physical) |
| **Extrapolation** | Often produces invalid shapes | Stays on manifold |
| **Computational Cost** | $O(D^2 N)$ for covariance | $O(K \cdot T \cdot M)$ per pair |
| **Correspondence** | Required | Optional (currents/varifolds) |
| **Implementation** | Straightforward | Complex, requires expertise |

### Detailed Comparison

#### Advantages of LDDMM over Linear PCA

1. **Topology Preservation**
   - PCA: Linear combinations can create self-intersecting surfaces
   - LDDMM: Diffeomorphisms guarantee valid surfaces

2. **Large Deformations**
   - PCA: Assumes small variations around mean
   - LDDMM: Handles arbitrary deformation magnitudes

3. **Proper Metric Structure**
   - PCA: Euclidean distance may not reflect anatomical similarity
   - LDDMM: Geodesic distance measures true shape difference

4. **Physical Plausibility**
   - PCA: Intermediate shapes may be unphysical
   - LDDMM: All interpolations are smooth deformations

5. **Correspondence-Free Matching**
   - PCA: Requires point-to-point correspondence
   - LDDMM: Can work with currents/varifolds without correspondence

#### Advantages of Linear PCA over LDDMM

1. **Computational Efficiency**
   - PCA: $O(D^2 N)$ for $N$ samples, $D$ dimensions
   - LDDMM: $O(K \cdot T \cdot D \cdot N)$ per pairwise registration, $K$ iterations

2. **Simplicity**
   - PCA: Standard linear algebra
   - LDDMM: Complex optimization, many hyperparameters

3. **Interpretability**
   - PCA: Principal components are intuitive
   - LDDMM: Initial momenta are abstract

4. **Robustness**
   - PCA: Closed-form solution, deterministic
   - LDDMM: Iterative optimization, may converge to local minima

5. **Data Requirements**
   - PCA: Works with any sample size $N > 1$
   - LDDMM: Requires careful parameter tuning, more sensitive to noise

### When to Use Each Method

**Use Linear PCA when:**
- Variations are small relative to shape size
- Topology violations are unlikely or acceptable
- Computational resources are limited
- Quick exploratory analysis is needed
- Data already has correspondence

**Use LDDMM when:**
- Large deformations are present
- Topology must be preserved (medical applications)
- Statistical analysis requires proper geometry
- Generating new samples must produce valid shapes
- Correspondence is unavailable or unreliable

---

## 8. Critical Analysis

### Strengths of LDDMM

1. **Mathematical Rigor**
   - Well-founded in differential geometry
   - Proper metric structure enables valid statistics
   - Geodesics provide optimal deformation paths

2. **Biological Validity**
   - Diffeomorphisms model realistic tissue transformations
   - No tearing, folding, or self-intersection
   - Consistent with anatomical constraints

3. **Generative Capability**
   - Can synthesize new valid shapes
   - Extrapolation along geodesics
   - Uncertainty quantification possible

### Limitations of LDDMM

1. **Computational Cost**
   - Registration: $O(K \cdot T \cdot D)$ per pair
   - Atlas building: $O(N^2)$ or $O(N)$ with approximations
   - GPU acceleration essential for 3D

2. **Hyperparameter Sensitivity**
   - Kernel width $\sigma$: Controls deformation scale
   - Regularization $\sigma_R$: Smoothness vs. matching trade-off
   - Time steps $T$: Accuracy vs. speed
   - Learning rate $\epsilon$: Convergence stability

3. **Local Minima**
   - Non-convex optimization
   - Results depend on initialization
   - Multi-scale strategies help but add complexity

4. **Scalability**
   - Pairwise registration doesn't scale to thousands of shapes
   - Memory requirements for high-resolution 3D data
   - Atlas building is iterative and slow

5. **Correspondence Limitation**
   - Current/varifold matching is approximate
   - Point-wise correspondence may still be needed for statistics
   - Surface parameterization challenges

### What Can We Do With LDDMM?

1. **Shape Registration**
   - Pairwise alignment of anatomical structures
   - Multi-modal registration (MRI to CT, etc.)
   - Longitudinal tracking of shape changes

2. **Statistical Analysis**
   - Population mean estimation (atlas)
   - Principal geodesic analysis (PGA)
   - Hypothesis testing on shape differences

3. **Shape Synthesis**
   - Generate new realistic shapes
   - Interpolate between shapes
   - Explore shape variability

4. **Disease Analysis**
   - Detect shape abnormalities
   - Track disease progression
   - Predict outcomes from shape features

---

## 9. Applications and Capabilities

### For Femur Modeling Project

**Immediate Applications:**
1. **Atlas Construction**: Build population mean femur
2. **Deformation Analysis**: Quantify inter-subject variability
3. **PGA**: Compute principal modes of shape variation
4. **Shape Synthesis**: Generate new plausible femur shapes

**Advanced Applications:**
1. **Pathology Detection**: Identify abnormal femurs (osteoarthritis, fractures)
2. **Implant Design**: Customize implants to patient anatomy
3. **Surgical Planning**: Predict post-operative shape
4. **Biomarker Discovery**: Shape features correlated with clinical outcomes

### Integration with Current Pipeline

```
Current Pipeline:
  Femur Data (VTK) → FemurDataset → PCA (linear) → Visualization

Enhanced Pipeline:
  Femur Data (VTK) → FemurDataset → LDDMM Atlas Building → PGA
                                  ↓
                          Initial Momenta {m₀ʲ}
                                  ↓
                    Tangent Space PCA → Principal Geodesics
                                  ↓
                    Shape Synthesis → Validation
```

---

## 10. References

### Foundational Papers

1. **Beg, M.F., Miller, M.I., Trouvé, A., & Younes, L. (2005)**. Computing large deformation metric mappings via geodesic flows of diffeomorphisms. *IJCV*, 61(2), 139-157.

2. **Miller, M.I., Trouvé, A., & Younes, L. (2002)**. On the metrics and Euler-Lagrange equations of computational anatomy. *Annual Review of Biomedical Engineering*, 4(1), 375-405.

3. **Younes, L. (2010)**. *Shapes and Diffeomorphisms*. Springer.

### Implementation References

4. **Vaillant, M., & Glaunès, J. (2005)**. Surface matching via currents. *IPMI*, 381-392.

5. **Durrleman, S., et al. (2009)**. Statistical models of sets of curves and surfaces based on currents. *Medical Image Analysis*, 13(5), 793-808.

6. **Tward, D., et al. (2020)**. Diffeomorphic registration with intensity transformation and missing data. *Frontiers in Neuroscience*, 14, 52.

### Software

7. **pyLDDMM**: Educational implementation - https://github.com/SteffenCzolbe/pyLDDMM
8. **emlddmm**: Production 3D registration - https://github.com/twardlab/emlddmm
9. **Deformetrica**: Currents/varifolds - https://www.deformetrica.org
10. **lagomorph**: PyTorch GPU implementation - https://github.com/jacobhinkle/lagomorph

---

*Document Version: 1.0*  
*Last Updated: 2024*
