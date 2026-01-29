# LDDMM Theory and Implementation Choices

This document explains the mathematical foundations of our LDDMM implementation
and justifies the algorithmic choices made.

## Table of Contents

1. [What is LDDMM?](#what-is-lddmm)
2. [The Shape Space Perspective](#the-shape-space-perspective)
3. [Geodesic Shooting](#geodesic-shooting)
4. [Energy Minimization and Registration](#energy-minimization-and-registration)
5. [The Kernel and RKHS](#the-kernel-and-rkhs)
6. [Atlas Building](#atlas-building)
7. [Tangent Space PCA](#tangent-space-pca)
8. [LDDMM vs Linear PCA: Key Differences](#lddmm-vs-linear-pca-key-differences)
9. [Implementation Choices](#implementation-choices)
10. [References](#references)

---

## What is LDDMM?

**Large Deformation Diffeomorphic Metric Mapping (LDDMM)** is a framework for
computing smooth, invertible transformations between shapes. It treats shapes
as points on a **Riemannian manifold**, not as vectors in Euclidean space.

### The Key Idea

Instead of directly computing a deformation φ: ℝ³ → ℝ³, LDDMM constructs φ as
the **flow of a time-varying velocity field** v(x, t):

$$
\frac{\partial \phi_t}{\partial t}(x) = v(\phi_t(x), t), \quad \phi_0 = \text{Id}
$$

This ensures:
1. **Diffeomorphism**: φ is smooth and invertible (no folding/tearing)
2. **Metric structure**: The "length" of the deformation path defines a distance

### The Fundamental Difference from Euclidean Methods

**LDDMM treats shapes as manifold-valued data**:

| Aspect | Euclidean (Linear PCA) | LDDMM |
|--------|------------------------|-------|
| **Shape space** | ℝ³ᴺ (flat vector space) | Diff(Ω) (curved manifold) |
| **Distance** | ‖S₁ - S₂‖ (Euclidean) | ∫₀¹ ‖vₜ‖²_V dt (geodesic) |
| **Mean** | Arithmetic: (1/K)Σᵢ Sᵢ | Fréchet: argmin Σᵢ d²(μ, Sᵢ) |
| **Interpolation** | Linear: (1-t)S₁ + tS₂ | Geodesic: Exp_S₁(t·Log_S₁(S₂)) |

The distance in LDDMM is **NOT** the Euclidean distance, even when shapes have
point correspondence. The geodesic distance measures the "cost" of the smoothest
diffeomorphism connecting two shapes, weighted by the kernel.

### Why Does This Matter?

For statistical shape analysis, we need:
- A **distance** between shapes (how different are two femurs?)
- A **mean** shape (what's the average femur?)
- A **linear space** for statistics (PCA requires vector operations)

LDDMM provides all three through its Riemannian geometry—but the mean and
distance are fundamentally different from their Euclidean counterparts.

---

## The Shape Space Perspective

### Shapes as Points on a Manifold

Consider the space of all possible femur shapes. This is not a vector space—you
can't simply add two femurs and get a valid femur. Instead, it's a **manifold**:
a curved space where linear operations don't directly apply.

```
        Shape Manifold (curved!)
           ╱╲
          ╱  ╲        Geodesic (shortest path on manifold)
     S₁ ●────● S₂     ← Shapes are points
          ╲  ╱        Straight line in Euclidean space
           ●            would cut through manifold!
          Atlas
```

### Why Not Just Use Euclidean Space?

Even with point correspondence, treating shapes as vectors in ℝ³ᴺ ignores the
**geometry of deformations**:

1. **Linear interpolation artifacts**: The midpoint (S₁ + S₂)/2 may have
   self-intersections or unphysical local stretching
2. **Wrong distance**: Euclidean distance treats all point movements equally;
   LDDMM penalizes non-smooth deformations more heavily
3. **Non-geodesic paths**: The straight line S₁ → S₂ is not the optimal
   deformation path on the shape manifold

### The Tangent Space

At any point (shape) on the manifold, there's a **tangent space**: a linear
approximation where we CAN do vector operations.

```
    Tangent Space at Atlas
    ━━━━━━━━━━━━━━━━━━━━━━
    │   →    →    →       │
    │  p₁   p₂   p₃  ...  │  ← Initial momenta (vectors)
    ━━━━━━━━━━━━━━━━━━━━━━
           ↑
         Atlas (base point)
```

The **initial momentum** p₀ at the atlas encodes "which direction and how far"
to shoot to reach a target shape.

**Critical point**: The momentum p₀ is NOT the displacement (target - source).
It's the initial velocity of the geodesic, transformed by the kernel.

---

## Geodesic Shooting

### The EPDiff Equation

LDDMM computes geodesics (shortest paths) on the shape manifold. The velocity
field v(t) evolves according to the **EPDiff equation**:

$$
\frac{\partial p}{\partial t} + \nabla_v p + (\nabla v)^T p + p \, \text{div}(v) = 0
$$

where p is the momentum (dual to velocity via the kernel: v = K * p).

### In Practice

Given initial momentum p₀ at the atlas:
1. Integrate EPDiff forward in time
2. The flow φ₁ deforms the atlas to the target shape
3. This is called **geodesic shooting**

### The Exponential Map

$$
\text{Exp}_\text{atlas}(p_0) = \phi_1(\text{atlas})
$$

The exponential map "shoots" from the atlas along the initial momentum to produce
a new shape.

### The Log Map

$$
\text{Log}_\text{atlas}(S) = p_0 \text{ such that } \text{Exp}_\text{atlas}(p_0) = S
$$

The log map finds the initial momentum that shoots to a given shape. This is
computed via **registration** (optimization)—it's NOT simply (S - atlas).

### The Geodesic Distance

The distance between two shapes is the length of the geodesic connecting them:

$$
d(S_1, S_2)^2 = \min_{v_t} \int_0^1 \|v_t\|_V^2 \, dt
$$

where v_t is the time-varying velocity field and ‖·‖_V is the RKHS norm defined
by the kernel. This is **fundamentally different** from Euclidean distance.

---

## Energy Minimization and Registration

Computing the geodesic distance—or equivalently, the log map—requires solving
an optimization problem. This section details the exact energy formulation and
numerical approach used in our implementation (via scikit-shapes).

### The Total Energy Functional

Given a source shape $S_{\text{source}}$ with vertices $\mathbf{q}_0 \in \mathbb{R}^{N \times 3}$
and a target shape $S_{\text{target}}$, LDDMM registration finds the initial
momentum $\mathbf{p}_0 \in \mathbb{R}^{N \times 3}$ by minimizing:

$$
E(\mathbf{p}_0) = \underbrace{\mathcal{L}_{\text{fid}}(\phi_1(S_{\text{source}}), S_{\text{target}})}_{\text{Fidelity term}} + \lambda \cdot \underbrace{H(\mathbf{p}_0, \mathbf{q}_0)}_{\text{Regularization term}}
$$

where:
- $\phi_1$ is the diffeomorphism at time $t=1$ generated by shooting from $\mathbf{p}_0$
- $\mathcal{L}_{\text{fid}}$ is the fidelity loss measuring correspondence quality
- $H$ is the Hamiltonian (kinetic energy of the deformation)
- $\lambda$ is the regularization weight balancing data fit vs. smoothness

### The Fidelity Term

For shapes with known point correspondence (our case), we use the **L2 loss**:

$$
\mathcal{L}_{\text{fid}}(\phi_1(S), T) = \sum_{i=1}^{N} \|\phi_1(\mathbf{q}_i) - \mathbf{t}_i\|^2
$$

where $\mathbf{q}_i$ are source vertices and $\mathbf{t}_i$ are corresponding target vertices.

This measures how closely the deformed source matches the target after applying
the diffeomorphism.

### The Regularization Term: Hamiltonian Formulation

The regularization is the **Hamiltonian** (kinetic energy) of the initial state:

$$
H(\mathbf{p}, \mathbf{q}) = \frac{1}{2} \langle \mathbf{p}, K_{\mathbf{q}} \mathbf{p} \rangle = \frac{1}{2} \sum_{i,j=1}^{N} \mathbf{p}_i^T K(\mathbf{q}_i, \mathbf{q}_j) \mathbf{p}_j
$$

where $K(\mathbf{q}_i, \mathbf{q}_j)$ is the kernel matrix evaluated between vertices.
With a Gaussian kernel:

$$
K(\mathbf{q}_i, \mathbf{q}_j) = \exp\left(-\frac{\|\mathbf{q}_i - \mathbf{q}_j\|^2}{2\sigma^2}\right)
$$

This quadratic form penalizes non-smooth deformations:
- Large momenta at isolated points → high energy (discouraged)
- Spatially coherent momenta → lower energy (encouraged)

The kernel scale $\sigma$ determines the correlation length: nearby points
(within distance $\sigma$) are encouraged to move together.

### The Optimization Space

The optimization is performed over the space of **initial momenta**:

$$
\mathbf{p}_0 \in \mathbb{R}^{N \times 3}
$$

This is a **finite-dimensional** space (one 3D vector per vertex), making the
problem tractable. The infinite-dimensional velocity field $v_t(x)$ for any
point $x \in \mathbb{R}^3$ is reconstructed as:

$$
v_t(x) = \sum_{i=1}^{N} K(x, \mathbf{q}_i(t)) \, \mathbf{p}_i(t)
$$

This parameterization by discrete momenta is the key insight that makes LDDMM
computationally feasible.

### Hamiltonian Dynamics: The Geodesic Equations

Given initial momentum $\mathbf{p}_0$, the shape and momentum evolve according
to **Hamilton's equations**:

$$
\dot{\mathbf{q}}_i = \frac{\partial H}{\partial \mathbf{p}_i} = \sum_{j=1}^{N} K(\mathbf{q}_i, \mathbf{q}_j) \mathbf{p}_j
$$

$$
\dot{\mathbf{p}}_i = -\frac{\partial H}{\partial \mathbf{q}_i}
$$

These ODEs are integrated from $t=0$ to $t=1$ using numerical solvers (Euler,
midpoint, or Runge-Kutta). The final positions $\mathbf{q}_1 = \phi_1(\mathbf{q}_0)$
give the morphed shape.

A crucial property of Hamiltonian systems is **energy conservation**:

$$
H(\mathbf{p}_t, \mathbf{q}_t) = H(\mathbf{p}_0, \mathbf{q}_0) \quad \forall t \in [0, 1]
$$

### Connection to Geodesic Distance

The geodesic distance is defined as the path length on the shape manifold:

$$
d(S_1, S_2)^2 = \min_{\{v_t\}} \int_0^1 \|v_t\|_V^2 \, dt
$$

where $\|v_t\|_V^2$ is the RKHS norm. For velocity fields parameterized by
momenta, this becomes:

$$
\|v_t\|_V^2 = 2 H(\mathbf{p}_t, \mathbf{q}_t)
$$

By energy conservation along geodesics:

$$
d(S_1, S_2)^2 = \int_0^1 2 H(\mathbf{p}_t, \mathbf{q}_t) \, dt = 2 H(\mathbf{p}_0, \mathbf{q}_0)
$$

Thus, **the squared geodesic distance equals twice the initial Hamiltonian**
of the optimal momentum. This is why minimizing the total energy
$E(\mathbf{p}_0) = \mathcal{L}_{\text{fid}} + \lambda H$ simultaneously:

1. Achieves correspondence (via fidelity term)
2. Finds the shortest geodesic (via Hamiltonian regularization)

### The Optimization Algorithm

We use **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), a
quasi-Newton method that:

1. Approximates the Hessian using gradient history (memory-efficient)
2. Converges superlinearly near the optimum
3. Is well-suited for smooth, high-dimensional problems like LDDMM

The gradient $\nabla_{\mathbf{p}_0} E$ is computed via automatic differentiation
through the ODE integration, leveraging PyTorch's `autograd` and the adjoint
method for efficiency.

### Summary of the Registration Pipeline

```
Input: Source shape S, Target shape T

1. Initialize: p₀ = 0 (zero momentum)

2. Repeat (L-BFGS iterations):
   a. Integrate Hamiltonian ODEs: (p₀, q₀) → (p₁, q₁)
   b. Compute fidelity: L_fid = ||q₁ - T||²
   c. Compute regularization: H = ½ ⟨p₀, K p₀⟩
   d. Total loss: E = L_fid + λ·H
   e. Backpropagate: compute ∇_{p₀} E
   f. Update: p₀ ← L-BFGS step

3. Output: Optimal momentum p₀* (the log map)
           Morphed shape q₁* = Exp_S(p₀*)
           Geodesic distance² ≈ 2·H(p₀*, q₀)
```

### Practical Considerations

**Regularization weight λ:**
- Small λ → tight data fit, potentially irregular deformation
- Large λ → smooth deformation, may not match target exactly
- We use λ = 0.01 as a default, allowing realistic anatomical variation

**Number of integration steps:**
- More steps → more accurate ODE integration, slower computation
- We use 5 steps as a balance between accuracy and speed

**Kernel scale σ:**
- Determines the spatial correlation of the deformation
- Should be ~10-20% of the shape's bounding box diagonal
- For femurs (~100mm): σ ≈ 10-15mm

---

## The Kernel and RKHS

### Why Regularization?

Without constraints, the velocity field v could be arbitrarily wild. We want
**smooth** deformations that preserve anatomical plausibility.

### The RKHS Framework

We constrain v to live in a **Reproducing Kernel Hilbert Space (RKHS)** V
defined by a kernel K:

$$
\|v\|_V^2 = \langle v, v \rangle_V = \int v(x)^T K^{-1}(x, y) v(y) \, dx \, dy
$$

### The Gaussian Kernel

We use a Gaussian kernel:

$$
K(x, y) = \exp\left(-\frac{\|x - y\|^2}{2\sigma^2}\right)
$$

**The scale parameter σ controls correlation:**
- **Small σ**: Local deformations, each point moves independently
- **Large σ**: Global deformations, nearby points move together

### Choosing σ for Femurs

Rule of thumb: σ ≈ 10-20% of the shape's bounding box diagonal.

For femurs (~100mm bounding box): σ ≈ 10-20mm.

---

## Atlas Building

### The Fréchet Mean

The atlas μ is the **Fréchet mean**: the shape that minimizes total squared
geodesic distance to all training shapes:

$$
\mu = \arg\min_S \sum_{i=1}^K d^2(S, S_i)
$$

where d is the **geodesic distance**, NOT the Euclidean distance.

### Why the Fréchet Mean ≠ Arithmetic Mean

Even with point correspondence, the Fréchet mean in LDDMM is NOT the arithmetic
mean. Here's why:

1. **Different distance metric**: The geodesic distance d(S₁, S₂) involves
   minimizing the integral ∫‖vₜ‖²_V dt over smooth velocity fields. This is
   NOT equal to ‖S₁ - S₂‖_F.

2. **Regularization matters**: The kernel K penalizes non-smooth deformations.
   Two shapes that differ by a smooth global rotation have smaller geodesic
   distance than two shapes differing by local jagged displacements of the
   same Euclidean magnitude.

3. **Curved geometry**: The minimizer of Σᵢ d²(μ, Sᵢ) on a curved manifold
   is generally NOT the same as (1/K) Σᵢ Sᵢ.

### Illustrative Example

Consider three 2D point clouds forming triangles:
- S₁: equilateral triangle at origin
- S₂: same triangle rotated 60°
- S₃: same triangle with one vertex displaced

Euclidean distance:
- d_E(S₁, S₂) could be large (all points moved)
- d_E(S₁, S₃) could be small (one point moved)

LDDMM geodesic distance:
- d_G(S₁, S₂) is small (smooth rotation)
- d_G(S₁, S₃) could be larger (local, less smooth deformation)

The Fréchet mean would weight these differently than arithmetic mean.

### Algorithm: Iterative Geodesic Averaging

```
1. Initialize: μ = arithmetic mean of shapes (approximation)
2. Repeat until convergence:
   a. Compute log maps: pᵢ = Log_μ(Sᵢ) for all shapes
      (via LDDMM registration, NOT displacement)
   b. Average in tangent space: p̄ = (1/K) Σᵢ pᵢ
   c. Update: μ ← Exp_μ(α · p̄)  where α ∈ (0, 1] is step size
```

### Our Implementation

We use true LDDMM geodesic averaging:

1. Initialize atlas as arithmetic mean of shapes (starting point only)
2. Compute log maps pᵢ = Log_μ(Sᵢ) via LDDMM registration
3. Average in tangent space: p̄ = (1/K) Σᵢ pᵢ
4. Update atlas: μ ← Exp_μ(α · p̄) via geodesic shooting
5. Repeat until convergence

This produces the true Fréchet mean on the shape manifold.

---

## Tangent Space PCA

### The Idea

Since the tangent space at the atlas IS a vector space, we can do standard PCA:

1. Compute all initial momenta: pᵢ = Log_atlas(Sᵢ) via LDDMM registration
2. Flatten to vectors: pᵢ ∈ ℝ^(N×3) → ℝ^(3N)
3. Subtract mean: p̃ᵢ = pᵢ - p̄
4. SVD: P̃ = UΣVᵀ
5. Principal components: columns of V (reshaped to N×3)

### Shape Synthesis

To generate a new shape from coefficients c = (c₁, ..., cₖ):

$$
p = \bar{p} + \sum_{j=1}^k c_j \cdot \sqrt{\lambda_j} \cdot v_j
$$

$$
S = \text{Exp}_\text{atlas}(p)
$$

where vⱼ are principal components, λⱼ are eigenvalues, and Exp is geodesic shooting.

### Shape Projection

To project a new shape S onto the PCA basis:

1. Compute log map: p = Log_atlas(S) via LDDMM registration
2. Center: p̃ = p - p̄
3. Project: c = p̃ · V (inner product with principal components)

### Interpretation

- **Component 1**: Direction of maximum variance (e.g., femur length)
- **Component 2**: Second-most variance (e.g., head angle)
- **Coefficients**: "Coordinates" in the shape space

---

## Why LDDMM Over Linear PCA?

### Key Differences

| Aspect | Linear PCA | LDDMM |
|--------|------------|-------|
| **Shape representation** | Vector in ℝ³ᴺ | Point on Diff(Ω) manifold |
| **Distance** | ‖S₁ - S₂‖_F | Geodesic (regularized path length) |
| **Mean** | (1/K) Σᵢ Sᵢ | Fréchet mean (iterative) |
| **"Momentum"** | Displacement: S - μ | Log map via registration |
| **Interpolation** | (1-t)S₁ + tS₂ | Geodesic shooting |
| **Extrapolation** | May self-intersect | Diffeomorphic (valid shapes) |

### Why Momenta ≠ Displacements

Linear PCA uses displacements: `momentum = target - source`

This ignores the geometry of deformations. True LDDMM momenta are obtained
by solving a registration problem that finds the smoothest diffeomorphism.

**The momentum p₀ satisfies**: v₀ = K * p₀ (velocity = kernel applied to momentum)

The kernel K enforces smoothness—nearby points must move coherently.

### Advantages of True LDDMM

1. **Topology preservation**: All interpolated/extrapolated shapes are valid
2. **Physically plausible deformations**: Smooth, coherent transformations
3. **Proper statistics**: Analysis respects the curved geometry of shape space
4. **Large deformation handling**: No artifacts from linear approximation
5. **Consistent distance metric**: Geodesic distance is geometrically meaningful

---

## Implementation Choices

### Why scikit-shapes?

| Criterion | emlddmm | scikit-shapes |
|-----------|---------|---------------|
| API design | Image-centric | Shape-native |
| Point cloud support | Hacky (dummy images) | Native |
| Documentation | Limited | Extensive |
| Maintenance | Stable but static | Active development |
| GPU support | Custom | KeOps (optimized) |

### Key Parameters

| Parameter | Our Choice | Rationale |
|-----------|------------|-----------|
| `n_steps=5` | 5 time steps | Balance of accuracy vs speed |
| `kernel="gaussian"` | Gaussian | Standard, well-understood |
| `scale=10-15mm` | ~10-15% of bounding box | Captures local+global variation |
| `regularization_weight=0.01` | Light regularization | Allows realistic variation |

### Computational Considerations

- **GPU acceleration**: scikit-shapes uses KeOps for fast kernel operations
- **Memory**: O(N²) for kernel matrix, but KeOps avoids explicit construction
- **Time complexity**: O(n_iter × n_steps × N²) per registration

---

## References

1. **Beg, M.F., et al.** (2005). "Computing Large Deformation Metric Mappings 
   via Geodesic Flows of Diffeomorphisms." *IJCV* 61(2), 139-157.

2. **Younes, L.** (2010). *Shapes and Diffeomorphisms*. Springer.

3. **Miller, M.I., Trouvé, A., & Younes, L.** (2015). "Hamiltonian Systems and 
   Optimal Control in Computational Anatomy." *Annual Review of Biomedical 
   Engineering* 17, 447-509.

4. **Durrleman, S., et al.** (2014). "Morphometry of anatomical shape complexes 
   with dense deformations and sparse parameters." *NeuroImage* 101, 35-49.

5. **scikit-shapes documentation**: https://scikit-shapes.github.io/
