# LDDMM Theory and Implementation Choices

This document explains the mathematical foundations of our LDDMM implementation
and justifies the algorithmic choices made.

## Table of Contents

1. [What is LDDMM?](#what-is-lddmm)
2. [The Shape Space Perspective](#the-shape-space-perspective)
3. [Geodesic Shooting](#geodesic-shooting)
4. [The Kernel and RKHS](#the-kernel-and-rkhs)
5. [Atlas Building](#atlas-building)
6. [Tangent Space PCA](#tangent-space-pca)
7. [Why Not Just Use Displacements?](#why-not-just-use-displacements)
8. [Implementation Choices](#implementation-choices)
9. [References](#references)

---

## What is LDDMM?

**Large Deformation Diffeomorphic Metric Mapping (LDDMM)** is a framework for
computing smooth, invertible transformations between shapes.

### The Key Idea

Instead of directly computing a deformation φ: ℝ³ → ℝ³, LDDMM constructs φ as
the **flow of a time-varying velocity field** v(x, t):

$$
\frac{\partial \phi_t}{\partial t}(x) = v(\phi_t(x), t), \quad \phi_0 = \text{Id}
$$

This ensures:
1. **Diffeomorphism**: φ is smooth and invertible (no folding/tearing)
2. **Metric structure**: The "length" of the deformation path defines a distance

### Why Does This Matter?

For statistical shape analysis, we need:
- A **distance** between shapes (how different are two femurs?)
- A **mean** shape (what's the average femur?)
- A **linear space** for statistics (PCA requires vector operations)

LDDMM provides all three through its Riemannian geometry.

---

## The Shape Space Perspective

### Shapes as Points on a Manifold

Consider the space of all possible femur shapes. This is not a vector space—you
can't simply add two femurs and get a valid femur. Instead, it's a **manifold**:
a curved space where linear operations don't directly apply.

```
        Shape Manifold
           ╱╲
          ╱  ╲
     S₁ ●────● S₂     ← Shapes are points
          ╲  ╱
           ●
          Atlas
```

### The Tangent Space

At any point (shape) on the manifold, there's a **tangent space**: a linear
approximation where we CAN do vector operations.

```
    Tangent Space at Atlas
    ━━━━━━━━━━━━━━━━━━━━━━
    │   →    →    →       │
    │  v₁   v₂   v₃  ...  │  ← Initial momenta (vectors)
    ━━━━━━━━━━━━━━━━━━━━━━
           ↑
         Atlas (base point)
```

The **initial momentum** p₀ at the atlas encodes "which direction and how far"
to shoot to reach a target shape.

---

## Geodesic Shooting

### The EPDiff Equation

LDDMM computes geodesics (shortest paths) on the shape manifold. The velocity
field v(t) evolves according to the **EPDiff equation**:

$$
\frac{\partial p}{\partial t} + \nabla_v p + (\nabla v)^T p + p \, \text{div}(v) = 0
$$

where p is the momentum (dual to velocity via the kernel).

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
computed via **registration** (optimization).

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

### Algorithm: Iterative Geodesic Averaging

```
1. Initialize: μ = arithmetic mean of shapes
2. Repeat until convergence:
   a. Compute log maps: pᵢ = Log_μ(Sᵢ) for all shapes
   b. Average in tangent space: p̄ = (1/K) Σᵢ pᵢ
   c. Update: μ ← Exp_μ(α · p̄)  where α ∈ (0, 1] is step size
```

### Special Case: Corresponding Points in Euclidean Space

**Important:** When shapes have **established point correspondence** and live
in **Euclidean space** (as our femur landmarks do), the Fréchet mean **equals
the arithmetic mean**.

#### Why This Is True

Consider K shapes S₁, ..., Sₖ where each Sᵢ ∈ ℝ^(N×3) represents N corresponding
landmark points.

1. **Distance is Euclidean**: With point correspondence, the geodesic distance
   between shapes reduces to:
   $$d(S_1, S_2) = \|S_1 - S_2\|_F$$
   (Frobenius norm = sum of squared point-to-point distances)

2. **Fréchet objective**: We minimize:
   $$E(\mu) = \sum_{i=1}^K \|\mu - S_i\|_F^2$$

3. **Solution**: Taking the gradient and setting to zero:
   $$\nabla_\mu E = 2\sum_{i=1}^K (\mu - S_i) = 0$$
   $$\Rightarrow \mu = \frac{1}{K}\sum_{i=1}^K S_i$$

This is exactly the **arithmetic mean**.

#### When Would Geodesic Averaging Differ?

- **No correspondence**: When matching is unknown, the distance involves
  optimization over correspondences, making it non-Euclidean
- **Manifold-valued data**: If points lie on a sphere or other manifold
- **Regularized matching**: If we penalize deformation complexity

**For our femur landmarks with known correspondence in ℝ³, arithmetic mean
is mathematically equivalent to the Fréchet mean.**

---

## Tangent Space PCA

### The Idea

Since the tangent space at the atlas IS a vector space, we can do standard PCA:

1. Compute all initial momenta: pᵢ = Log_atlas(Sᵢ)
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

where vⱼ are principal components and λⱼ are eigenvalues.

### Interpretation

- **Component 1**: Direction of maximum variance (e.g., femur length)
- **Component 2**: Second-most variance (e.g., head angle)
- **Coefficients**: "Coordinates" in the shape space

---

## Why Not Just Use Displacements?

The previous implementation approximated:

```python
momentum = target - source  # WRONG for true LDDMM
```

### When This Works

This is valid in the **linearized small-deformation regime**:
- Deformations are infinitesimally small
- The kernel is identity (no regularization)
- We're at the limit: LDDMM → linear elasticity

### When This Fails

For realistic anatomical variation:
- **Non-linearity**: Large rotations, scaling require geodesics
- **Regularization ignored**: No smoothness constraint on deformation
- **Geodesic distance wrong**: Euclidean ≠ geodesic for large deformations

### Practical Impact

| Aspect | Displacement | True LDDMM |
|--------|--------------|------------|
| Reconstruction error | Higher for large deformations | Lower |
| Interpolation | May produce invalid shapes | Stays on manifold |
| Geodesic distance | Approximation only | Exact |
| Statistical validity | Questionable | Principled |

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
