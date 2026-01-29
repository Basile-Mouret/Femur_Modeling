# LDDMM Pipeline for Statistical Shape Analysis

**A Self-Contained Introduction to Large Deformation Diffeomorphic Metric Mapping with Point Landmarks**

---

## Table of Contents

1. [Introduction](#1-introduction)
2. [The Shape Space Framework](#2-the-shape-space-framework)
   - 2.1 [Why Not Euclidean Space?](#21-why-not-euclidean-space)
   - 2.2 [The Diffeomorphism Group as Shape Space](#22-the-diffeomorphism-group-as-shape-space)
   - 2.3 [Riemannian Structure and Geodesics](#23-riemannian-structure-and-geodesics)
3. [The RKHS Framework and Kernel Methods](#3-the-rkhs-framework-and-kernel-methods)
   - 3.1 [The Reproducing Kernel Hilbert Space](#31-the-reproducing-kernel-hilbert-space)
   - 3.2 [Finite-Dimensional Reduction via Landmarks](#32-finite-dimensional-reduction-via-landmarks)
   - 3.3 [The Gaussian Kernel](#33-the-gaussian-kernel)
4. [Geodesic Distance and Energy Minimization](#4-geodesic-distance-and-energy-minimization)
   - 4.1 [The Variational Problem](#41-the-variational-problem)
   - 4.2 [The Hamiltonian Formulation](#42-the-hamiltonian-formulation)
   - 4.3 [Conservation of the Hamiltonian](#43-conservation-of-the-hamiltonian)
   - 4.4 [Connection to Geodesic Distance](#44-connection-to-geodesic-distance)
5. [Numerical Implementation](#5-numerical-implementation)
   - 5.1 [The Registration Problem](#51-the-registration-problem)
   - 5.2 [ODE Integration for Geodesic Shooting](#52-ode-integration-for-geodesic-shooting)
   - 5.3 [The LBFGS Optimizer](#53-the-lbfgs-optimizer)
   - 5.4 [Complete Registration Algorithm](#54-complete-registration-algorithm)
6. [Atlas Building: The Fréchet Mean](#6-atlas-building-the-fréchet-mean)
   - 6.1 [Definition and Motivation](#61-definition-and-motivation)
   - 6.2 [Iterative Geodesic Averaging Algorithm](#62-iterative-geodesic-averaging-algorithm)
7. [Tangent Space PCA](#7-tangent-space-pca)
   - 7.1 [The Tangent Space at the Atlas](#71-the-tangent-space-at-the-atlas)
   - 7.2 [PCA in Tangent Space](#72-pca-in-tangent-space)
   - 7.3 [Shape Synthesis and Projection](#73-shape-synthesis-and-projection)
8. [Parameter Selection Guidelines](#8-parameter-selection-guidelines)
9. [References](#9-references)

---

## 1. Introduction

This document presents a rigorous yet accessible introduction to **Large Deformation Diffeomorphic Metric Mapping (LDDMM)** for statistical shape analysis, specifically in the context of point landmarks with known correspondence—such as anatomical structures (e.g., femur meshes) where each vertex corresponds across subjects.

Our goal is to build a **statistical shape model** that can:
1. Compute a **mean shape** (atlas) that is geometrically meaningful
2. Identify **principal modes of variation** in a population
3. **Synthesize** new plausible shapes by sampling from the learned distribution
4. **Project** new shapes onto the learned basis for classification or analysis

The key insight of LDDMM is that shapes live on a **curved manifold**, not in a flat Euclidean space. This has profound implications for how we define distances, means, and perform statistics.

**Notation conventions:**
- Shapes are point clouds $\mathbf{q} = (\mathbf{q}_1, \ldots, \mathbf{q}_N) \in \mathbb{R}^{N \times 3}$
- Momenta are vector fields at the landmarks $\mathbf{p} = (\mathbf{p}_1, \ldots, \mathbf{p}_N) \in \mathbb{R}^{N \times 3}$
- $K$ denotes the kernel matrix, $\sigma$ the kernel scale
- $\phi_t$ denotes the diffeomorphism at time $t$
- $H$ denotes the Hamiltonian (kinetic energy)

---

## 2. The Shape Space Framework

### 2.1 Why Not Euclidean Space?

A natural first approach to shape analysis would be to treat each shape as a vector in $\mathbb{R}^{3N}$ (concatenating all vertex coordinates) and apply standard Euclidean statistics. However, this approach has fundamental limitations:

**Problem 1: The Euclidean mean may be invalid.**
Consider two shapes that differ by a rotation. Their Euclidean midpoint
$$\mathbf{q}_{\text{mid}} = \frac{1}{2}(\mathbf{q}_1 + \mathbf{q}_2)$$
may exhibit self-intersections or anatomically impossible configurations.

**Problem 2: Euclidean distance ignores deformation geometry.**
Two shapes differing by a smooth global rotation may have large Euclidean distance, while two shapes differing by local jagged displacements may have small Euclidean distance. Yet the former is a "simpler" transformation.

**Problem 3: Linear interpolation leaves the manifold.**
The straight-line path from $\mathbf{q}_1$ to $\mathbf{q}_2$ in $\mathbb{R}^{3N}$ does not correspond to a smooth, invertible deformation of space.

These issues motivate the **Riemannian approach**: treating shapes as points on a curved manifold equipped with a metric that reflects the geometry of smooth deformations.

### 2.2 The Diffeomorphism Group as Shape Space

Following the framework of Grenander [13] and developed by Miller, Trouvé, and Younes [24, 35], we model shapes as elements of an **orbit** under the action of diffeomorphisms.

**Definition (Diffeomorphism).** A diffeomorphism $\phi: \mathbb{R}^3 \to \mathbb{R}^3$ is a smooth, invertible map with smooth inverse. The set of all such maps forms a group $\text{Diff}(\mathbb{R}^3)$.

**Definition (Shape Orbit).** Given a template shape $\mathbf{q}_{\text{temp}}$, the orbit
$$\mathcal{O} = \{\phi \cdot \mathbf{q}_{\text{temp}} : \phi \in G\}$$
is the set of all shapes obtainable by diffeomorphic deformation. Here $G \subset \text{Diff}(\mathbb{R}^3)$ is a suitable subgroup.

The key idea is to construct diffeomorphisms as **flows of time-varying velocity fields**. Given a velocity field $v: [0,1] \times \mathbb{R}^3 \to \mathbb{R}^3$, the flow $\phi_t$ satisfies:
$$\frac{\partial \phi_t}{\partial t}(x) = v_t(\phi_t(x)), \quad \phi_0 = \text{Id}$$

This construction guarantees that:
1. $\phi_1$ is a diffeomorphism (if $v$ is sufficiently smooth)
2. The transformation is invertible—no folding or tearing
3. Intermediate shapes $\phi_t(\mathbf{q})$ are all valid shapes

### 2.3 Riemannian Structure and Geodesics

To do statistics on the shape manifold, we need a notion of **distance**. Our ultimate goal is to compute the **Fréchet mean** (atlas) of a population of shapes $\{S_1, \ldots, S_K\}$:
$$\mu^* = \arg\min_{\mu} \sum_{k=1}^{K} d(\mu, S_k)^2$$

This requires computing geodesic distances $d(\mu, S_k)$ between shapes. The following sections develop the mathematical machinery to make these computations tractable.

#### The Riemannian Metric

A **Riemannian metric** on a manifold $\mathcal{M}$ assigns to each point $q \in \mathcal{M}$ an inner product $\langle \cdot, \cdot \rangle_q$ on the tangent space $T_q\mathcal{M}$, varying smoothly with $q$ [21].

In LDDMM, shapes are identified with diffeomorphisms (modulo a template), so the tangent space at a shape $\mathbf{q}$ consists of **velocity fields** $v: \mathbb{R}^3 \to \mathbb{R}^3$. The Riemannian metric is defined via a norm $\|\cdot\|_V$ on velocity fields:
$$\langle v, w \rangle_{\mathbf{q}} = \langle v, w \rangle_V$$

where $\langle \cdot, \cdot \rangle_V$ is the inner product of a **Reproducing Kernel Hilbert Space** (RKHS), detailed in Section 3. This choice of metric—encoding smoothness constraints via a kernel—is what makes LDDMM fundamentally different from Euclidean approaches.

**Key point:** We are indeed constructing a proper Riemannian manifold. However, in practice we work primarily with **geodesics** (shortest paths) rather than manipulating the metric directly. The metric's role is to define what "shortest" means.

#### Path Length and Geodesic Distance

**Definition (Path Length).** For a path of velocity fields $\{v_t\}_{t \in [0,1]}$, the length is:
$$L[v] = \int_0^1 \|v_t\|_V \, dt$$

where $\|\cdot\|_V$ is the RKHS norm (Section 3.1).

**Definition (Geodesic Distance).** The distance between shapes $S_1$ and $S_2$ is:
$$d(S_1, S_2) = \inf_{\{v_t\}} L[v]$$

where the infimum is over all velocity fields whose flow maps $S_1$ to $S_2$.

The **geodesic** is the path achieving this minimum. Like straight lines in Euclidean space, geodesics are curves of zero acceleration in the sense of the Riemannian connection [21].

#### The Exponential and Logarithm Maps

Two fundamental operations emerge from this structure:

- **Exponential map** $\text{Exp}_\mu(\mathbf{p})$: Starting at shape $\mu$, "shoot" along the geodesic with initial velocity $\mathbf{p}$ to reach a new shape. This is **geodesic shooting**.

- **Logarithm map** $\text{Log}_\mu(S)$: Find the initial velocity $\mathbf{p}$ such that shooting from $\mu$ reaches $S$. This is **registration**.

These maps are inverses: $\text{Exp}_\mu(\text{Log}_\mu(S)) = S$. Computing the atlas requires both: Log maps to project shapes to the tangent space, and Exp maps to update the atlas estimate.

---

## 3. The RKHS Framework and Kernel Methods

### 3.1 The Reproducing Kernel Hilbert Space

Without constraints, velocity fields could be arbitrarily irregular, leading to non-smooth or non-invertible transformations. We constrain $v$ to lie in a **Reproducing Kernel Hilbert Space (RKHS)** $V$ defined by a positive-definite kernel $K$.

**Definition (RKHS Norm).** For a kernel $K: \mathbb{R}^3 \times \mathbb{R}^3 \to \mathbb{R}$, the RKHS norm is:
$$\|v\|_V^2 = \int_{\mathbb{R}^3 \times \mathbb{R}^3} v(x)^T K^{-1}(x, y) v(y) \, dx \, dy$$

This norm penalizes non-smooth velocity fields: the smoother the kernel $K$, the smoother the allowed deformations.

**Reproducing Property.** A key feature of RKHS is that point evaluations are continuous:
$$v(x) = \langle v, K(x, \cdot) \rangle_V$$

This allows us to represent velocity fields via their values at discrete points.

### 3.2 Finite-Dimensional Reduction via Landmarks

For shapes represented by $N$ landmark points (our case), a fundamental result [17, 24] shows that the optimal velocity field can be written as:
$$v_t(x) = \sum_{i=1}^{N} K(x, \mathbf{q}_i(t)) \, \mathbf{p}_i(t)$$

where:
- $\mathbf{q}_i(t)$ are the landmark positions at time $t$
- $\mathbf{p}_i(t)$ are the **momenta** at the landmarks
- $K(x, y)$ is the kernel evaluated at positions $x$ and $y$

**This is the key insight that makes LDDMM computationally tractable**: instead of optimizing over infinite-dimensional velocity fields, we optimize over finite-dimensional initial momenta $\mathbf{p}_0 \in \mathbb{R}^{N \times 3}$.

### 3.3 The Gaussian Kernel

We use the **Gaussian kernel**:
$$K(\mathbf{q}_i, \mathbf{q}_j) = \exp\left(-\frac{\|\mathbf{q}_i - \mathbf{q}_j\|^2}{2\sigma^2}\right)$$

**Interpretation of the scale parameter $\sigma$:**
- $\sigma$ determines the **correlation length** of the deformation
- Points within distance $\sigma$ tend to move together
- Points farther than $\sigma$ can move more independently

**Choosing $\sigma$:**
- Too small: deformation is too local, may not capture global variation
- Too large: deformation is too rigid, loses local detail
- Rule of thumb: $\sigma \approx 10$–$20\%$ of the shape's bounding box diagonal

For femur meshes with bounding box diagonal $\approx 100$ mm, we use $\sigma \approx 10$–$15$ mm.

---

## 4. Geodesic Distance and Energy Minimization

### 4.1 The Variational Problem

The geodesic distance between shapes $S_1$ (source) and $S_2$ (target) is defined as:
$$d(S_1, S_2)^2 = \min_{\{v_t\}} \int_0^1 \|v_t\|_V^2 \, dt$$

subject to the constraint that the flow $\phi_1$ generated by $\{v_t\}$ maps $S_1$ to $S_2$.

Using the landmark parameterization, the RKHS norm becomes [17, 24]:
$$\|v_t\|_V^2 = \sum_{i,j=1}^{N} \mathbf{p}_i(t)^T K(\mathbf{q}_i(t), \mathbf{q}_j(t)) \mathbf{p}_j(t) = 2H(\mathbf{p}_t, \mathbf{q}_t)$$

where $H$ is the **Hamiltonian** (see next section).

### 4.2 The Hamiltonian Formulation

The dynamics of geodesics in LDDMM follow **Hamiltonian mechanics** [1, 24]. The Hamiltonian function is:
$$H(\mathbf{p}, \mathbf{q}) = \frac{1}{2} \sum_{i,j=1}^{N} \mathbf{p}_i^T K(\mathbf{q}_i, \mathbf{q}_j) \mathbf{p}_j = \frac{1}{2} \langle \mathbf{p}, K_{\mathbf{q}} \mathbf{p} \rangle$$

where $K_{\mathbf{q}}$ is the $N \times N$ kernel matrix with entries $K(\mathbf{q}_i, \mathbf{q}_j)$.

**Hamilton's equations** govern the evolution of positions and momenta along geodesics:
$$\dot{\mathbf{q}}_i = \frac{\partial H}{\partial \mathbf{p}_i} = \sum_{j=1}^{N} K(\mathbf{q}_i, \mathbf{q}_j) \mathbf{p}_j$$

$$\dot{\mathbf{p}}_i = -\frac{\partial H}{\partial \mathbf{q}_i}$$

The second equation involves the gradient of the kernel:
$$\dot{\mathbf{p}}_i = -\sum_{j=1}^{N} \nabla_{\mathbf{q}_i} K(\mathbf{q}_i, \mathbf{q}_j) (\mathbf{p}_i^T \mathbf{p}_j)$$

For the Gaussian kernel, this gradient is:
$$\nabla_{\mathbf{q}_i} K(\mathbf{q}_i, \mathbf{q}_j) = -\frac{1}{\sigma^2} K(\mathbf{q}_i, \mathbf{q}_j) (\mathbf{q}_i - \mathbf{q}_j)$$

### 4.3 Conservation of the Hamiltonian

A fundamental property of Hamiltonian systems is **energy conservation** [1]:
$$H(\mathbf{p}_t, \mathbf{q}_t) = H(\mathbf{p}_0, \mathbf{q}_0), \quad \forall t \in [0, 1]$$

**Proof sketch:** Along a solution of Hamilton's equations,
$$\frac{dH}{dt} = \sum_i \left(\frac{\partial H}{\partial \mathbf{q}_i} \dot{\mathbf{q}}_i + \frac{\partial H}{\partial \mathbf{p}_i} \dot{\mathbf{p}}_i\right) = \sum_i \left(\frac{\partial H}{\partial \mathbf{q}_i} \frac{\partial H}{\partial \mathbf{p}_i} - \frac{\partial H}{\partial \mathbf{p}_i} \frac{\partial H}{\partial \mathbf{q}_i}\right) = 0$$

**Physical intuition:** The Hamiltonian $H$ represents the **kinetic energy** of the deformation. Conservation means that a geodesic maintains constant "speed"—like a ball rolling on a frictionless surface. The deformation neither accelerates nor decelerates; it flows at constant energy.

**Geometric intuition:** In Riemannian geometry, geodesics are curves of constant speed (when parameterized by arc length). The Hamiltonian being constant is the LDDMM manifestation of this principle.

**Computational consequence:** This conservation law is the cornerstone of **geodesic shooting**: the entire geodesic is determined by the initial conditions $(\mathbf{p}_0, \mathbf{q}_0)$. We don't need to store the full path—just the initial momentum $\mathbf{p}_0$.

### 4.4 Connection to Geodesic Distance

The geodesic distance can now be expressed in terms of the initial Hamiltonian:
$$d(S_1, S_2)^2 = \int_0^1 2H(\mathbf{p}_t, \mathbf{q}_t) \, dt = 2H(\mathbf{p}_0, \mathbf{q}_0) \cdot 1 = 2H(\mathbf{p}_0^*, \mathbf{q}_0)$$

where $\mathbf{p}_0^*$ is the **optimal initial momentum** that generates the geodesic from $S_1$ to $S_2$.

**Why this is remarkable:** We have reduced an infinite-dimensional problem (finding the optimal path through the space of diffeomorphisms) to a finite-dimensional one (finding the initial momentum $\mathbf{p}_0 \in \mathbb{R}^{N \times 3}$). The entire geodesic—and thus the distance—is encoded in this single vector field at the source shape.

**Key insight:** Finding the geodesic distance reduces to finding the initial momentum $\mathbf{p}_0$ that:
1. Generates a geodesic hitting the target $S_2$ at time $t=1$
2. Has minimal Hamiltonian $H(\mathbf{p}_0, \mathbf{q}_0)$

This is the **registration problem**.

#### Translation to Shape Space Vocabulary

The Hamiltonian formulation connects directly to the Riemannian operations introduced in Section 2.3:

| Riemannian Concept | Hamiltonian Formulation |
|--------------------|-------------------------|
| Tangent vector at $\mu$ | Initial momentum $\mathbf{p}_0$ |
| Geodesic from $\mu$ to $S$ | Solution of Hamilton's equations from $(\mathbf{p}_0, \mu)$ |
| Exponential map $\text{Exp}_\mu(\mathbf{p}_0)$ | Integrate Hamilton's equations, return $\mathbf{q}_1$ |
| Logarithm map $\text{Log}_\mu(S)$ | Find $\mathbf{p}_0$ such that $\mathbf{q}_1 = S$ (registration) |
| Geodesic distance $d(\mu, S)$ | $\sqrt{2H(\mathbf{p}_0^*, \mu)}$ where $\mathbf{p}_0^* = \text{Log}_\mu(S)$ |
| Riemannian metric $\langle \mathbf{p}, \mathbf{p}' \rangle_\mu$ | $\mathbf{p}^T K_\mu \mathbf{p}'$ (kernel inner product) |

**The registration problem is computing the Log map.** Given source $\mu$ and target $S$, we seek the initial momentum $\mathbf{p}_0$ that "shoots" from $\mu$ to $S$. This momentum is the tangent vector pointing from $\mu$ toward $S$ on the shape manifold.

---

## 5. Numerical Implementation

### 5.1 The Registration Problem

In practice, we cannot enforce the hard constraint $\phi_1(S_1) = S_2$ exactly (numerical noise, different mesh topologies, etc.). Instead, we solve a **soft registration** problem:

**Minimize over $\mathbf{p}_0$:**
$$E(\mathbf{p}_0) = \underbrace{\sum_{i=1}^{N} \|\phi_1(\mathbf{q}_i^{\text{src}}) - \mathbf{q}_i^{\text{tgt}}\|^2}_{\text{Fidelity term } \mathcal{L}_{\text{fid}}} + \lambda \cdot \underbrace{H(\mathbf{p}_0, \mathbf{q}_0)}_{\text{Regularization term}}$$

where:
- **Fidelity term**: L2 distance between morphed source and target (requires correspondence)
- **Regularization term**: Hamiltonian enforcing smooth deformation
- $\lambda > 0$: trade-off parameter

The regularization weight $\lambda$ balances:
- Small $\lambda$: tight fit to target, potentially irregular deformation
- Large $\lambda$: smooth deformation, may not match target exactly

### 5.2 ODE Integration for Geodesic Shooting

To evaluate $E(\mathbf{p}_0)$, we need to compute $\phi_1(\mathbf{q}_0)$. This requires **integrating Hamilton's equations** from $t=0$ to $t=1$:

```
Input: Initial positions q₀, initial momentum p₀
Output: Final positions q₁

1. Set (p, q) = (p₀, q₀)
2. For t = 0, Δt, 2Δt, ..., 1-Δt:
   a. Compute K = kernel matrix K(qᵢ, qⱼ)
   b. Compute q̇ = K @ p                    // Hamilton's first equation
   c. Compute ṗ = -∇_q H(p, q)             // Hamilton's second equation  
   d. Update: (p, q) ← (p + Δt·ṗ, q + Δt·q̇)
3. Return q
```

**Numerical solvers:**
- **Euler**: First-order, simple but requires small $\Delta t$
- **Midpoint**: Second-order, better accuracy
- **Runge-Kutta 4 (RK4)**: Fourth-order, most accurate

Our implementation uses **scikit-shapes** [32], which employs `torchdiffeq` [7] for differentiable ODE solving with automatic differentiation through the integration.

**Number of steps $n_{\text{steps}}$:**
- This discretizes the time interval $[0, 1]$ into $n_{\text{steps}}$ sub-intervals
- More steps → more accurate geodesic, but slower
- Typical choice: $n_{\text{steps}} = 3$–$10$

### 5.3 The LBFGS Optimizer

To minimize $E(\mathbf{p}_0)$, we use **L-BFGS** (Limited-memory Broyden–Fletcher–Goldfarb–Shanno) [26], a quasi-Newton method that:

1. Approximates the Hessian using a history of gradients (memory-efficient)
2. Achieves superlinear convergence near the optimum
3. Is well-suited for smooth, high-dimensional problems

**Gradient computation:**
The gradient $\nabla_{\mathbf{p}_0} E$ is computed via **automatic differentiation** through the ODE integration. This is the key advantage of using a differentiable ODE solver: we can backpropagate through the entire shooting procedure.

### 5.4 Complete Registration Algorithm

Putting it all together:

```
REGISTRATION(source, target, config)
────────────────────────────────────────────────────────────
Input:  source  ∈ ℝ^{N×3}  // Source shape
        target  ∈ ℝ^{N×3}  // Target shape
        config             // Parameters: σ, λ, n_steps, n_iter

Output: p₀*     ∈ ℝ^{N×3}  // Optimal initial momentum
        morphed ∈ ℝ^{N×3}  // Morphed source ≈ target

1. Initialize: p₀ ← 0

2. For iter = 1, 2, ..., n_iter (LBFGS iterations):
   
   a. GEODESIC SHOOTING:
      • Integrate Hamilton's equations from (p₀, source) to get q₁
      • This involves n_steps ODE integration steps
   
   b. COMPUTE LOSS:
      • L_fid = Σᵢ ||q₁,ᵢ - targetᵢ||²
      • H     = ½ · p₀ᵀ K(source) p₀
      • E     = L_fid + λ·H
   
   c. BACKPROPAGATE:
      • Compute ∇_{p₀} E via autodiff through ODE
   
   d. UPDATE:
      • p₀ ← LBFGS_step(p₀, ∇_{p₀} E)

3. Return p₀*, q₁
────────────────────────────────────────────────────────────
```

**Computational cost per registration:**
- $O(n_{\text{iter}} \times n_{\text{steps}} \times N^2)$ for kernel evaluations
- In practice: seconds to minutes per pair, depending on $N$

---

## 6. Atlas Building: The Fréchet Mean

### 6.1 Definition and Motivation

Given a population of shapes $\{S_1, \ldots, S_K\}$, we seek a **mean shape** that is central to the population. In Riemannian geometry, the appropriate notion is the **Fréchet mean** [12]:

$$\mu^* = \arg\min_{\mu} \sum_{k=1}^{K} d(\mu, S_k)^2$$

where $d$ is the **geodesic distance**, not the Euclidean distance.

**Why not use the arithmetic mean?**
Even with point correspondence, the Fréchet mean differs from the arithmetic mean because:
1. The geodesic distance $d(S_1, S_2)$ differs from Euclidean distance $\|S_1 - S_2\|$
2. The minimizer of $\sum_k d(\mu, S_k)^2$ on a curved manifold differs from $\frac{1}{K}\sum_k S_k$

**Example:** Consider shapes differing by smooth rotations vs. local jagged displacements. LDDMM penalizes irregular deformations more heavily, so the Fréchet mean accounts for deformation smoothness.

### 6.2 Iterative Geodesic Averaging Algorithm

The Fréchet mean is computed via **iterative geodesic averaging** [10, 18]:

```
ATLAS_BUILDING(shapes, config, max_iter, tol, α)
────────────────────────────────────────────────────────────
Input:  shapes = {S₁, ..., Sₖ}   // K shapes, each ∈ ℝ^{N×3}
        config                    // LDDMM parameters
        max_iter                  // Maximum iterations
        tol                       // Convergence tolerance
        α ∈ (0, 1]               // Step size

Output: μ                         // Fréchet mean (atlas)
        {p₁, ..., pₖ}            // Momenta from atlas to each shape

1. Initialize: μ ← (1/K) Σₖ Sₖ  // Arithmetic mean as starting point

2. For iter = 1, 2, ..., max_iter:
   
   a. COMPUTE LOG MAPS (registration):
      For k = 1, ..., K:
        pₖ ← Log_μ(Sₖ)          // LDDMM registration: μ → Sₖ
                                 // pₖ is the initial momentum
   
   b. AVERAGE IN TANGENT SPACE:
      p̄ ← (1/K) Σₖ pₖ
   
   c. UPDATE ATLAS (exponential map):
      μ ← Exp_μ(α · p̄)          // Geodesic shooting with momentum α·p̄
   
   d. CHECK CONVERGENCE:
      energy ← Σₖ H(pₖ, μ)
      If |energy - prev_energy| / |prev_energy| < tol:
        Break

3. Final momenta computation:
   For k = 1, ..., K:
     pₖ ← Log_μ(Sₖ)              // Recompute at converged atlas

4. Return μ, {p₁, ..., pₖ}
────────────────────────────────────────────────────────────
```

**Key operations:**
- **Log map** $\text{Log}_\mu(S)$: Find momentum $\mathbf{p}$ such that shooting from $\mu$ with $\mathbf{p}$ reaches $S$. This is LDDMM registration.
- **Exp map** $\text{Exp}_\mu(\mathbf{p})$: Shoot from $\mu$ with momentum $\mathbf{p}$ to get a new shape. This is geodesic shooting.

**Convergence:**
The algorithm typically converges in 5–15 iterations. The step size $\alpha < 1$ can improve stability for difficult populations.

---

## 7. Tangent Space PCA

### 7.1 The Tangent Space at the Atlas

The **tangent space** $T_\mu \mathcal{M}$ at the atlas $\mu$ is the linearization of the shape manifold at that point. It is a **vector space** where standard linear algebra applies.

**Geometric intuition:**
```
           Shape Manifold (curved)
              ╱╲
             ╱  ╲
        S₁ ●    ● S₂
             ╲  ╱
              ●  ← Atlas μ
             ╱|╲
       ─────────────  Tangent space T_μ M (flat)
         p₁  0  p₂   ← Momenta (tangent vectors)
```

Each shape $S_k$ corresponds to a tangent vector $\mathbf{p}_k = \text{Log}_\mu(S_k)$—the initial momentum that shoots from $\mu$ to $S_k$.

**Critical distinction:**
The momentum $\mathbf{p}_k$ is **not** the displacement $S_k - \mu$. It is obtained by solving the registration optimization problem, which accounts for the geometry of smooth deformations.

### 7.2 PCA in Tangent Space

Since the tangent space is linear, we can apply **standard PCA** to the momenta $\{\mathbf{p}_1, \ldots, \mathbf{p}_K\}$:

```
TANGENT_PCA(atlas, momenta, n_components)
────────────────────────────────────────────────────────────
Input:  atlas   ∈ ℝ^{N×3}         // Fréchet mean
        momenta ∈ ℝ^{K×N×3}       // K momenta, each N×3
        n_components              // Number of PCs to keep

Output: mean_momentum ∈ ℝ^{N×3}
        components    ∈ ℝ^{n_comp×N×3}  // Principal components
        eigenvalues   ∈ ℝ^{n_comp}       // Variances

1. Compute mean momentum:
   p̄ ← (1/K) Σₖ pₖ

2. Center the data:
   p̃ₖ ← pₖ - p̄   for k = 1, ..., K

3. Flatten to matrix:
   M ∈ ℝ^{K×3N}  where row k is flatten(p̃ₖ)

4. SVD:
   M = U Σ Vᵀ

5. Extract components:
   • eigenvalues = σᵢ² / (K-1)
   • components = first n_components rows of Vᵀ, reshaped to N×3

6. Return mean_momentum, components, eigenvalues
────────────────────────────────────────────────────────────
```

**Interpretation:**
- **Component $v_j$**: Direction of $j$-th largest variance in momentum space
- **Eigenvalue $\lambda_j$**: Variance along component $j$
- **Explained variance**: $\lambda_j / \sum_i \lambda_i$

### 7.3 Shape Synthesis and Projection

**Synthesis** (generating new shapes):
Given PCA coefficients $\mathbf{c} = (c_1, \ldots, c_n)$:

1. Construct momentum:
   $$\mathbf{p} = \bar{\mathbf{p}} + \sum_{j=1}^{n} c_j \sqrt{\lambda_j} \, \mathbf{v}_j$$

2. Shoot from atlas:
   $$S_{\text{new}} = \text{Exp}_\mu(\mathbf{p})$$

**Projection** (analyzing new shapes):
Given a new shape $S$:

1. Compute log map (registration):
   $$\mathbf{p} = \text{Log}_\mu(S)$$

2. Center:
   $$\tilde{\mathbf{p}} = \mathbf{p} - \bar{\mathbf{p}}$$

3. Project onto components:
   $$c_j = \langle \tilde{\mathbf{p}}, \mathbf{v}_j \rangle / \sqrt{\lambda_j}$$

**Note:** Both synthesis and projection use **true LDDMM** (geodesic shooting for Exp, registration for Log), not linear approximations.

---

## 8. Parameter Selection Guidelines

| Parameter | Symbol | Typical Range | Effect |
|-----------|--------|---------------|--------|
| **Kernel scale** | $\sigma$ | 10–20% of bbox | Spatial correlation of deformation |
| **Regularization weight** | $\lambda$ | 0.001–0.1 | Smoothness vs. data fit |
| **ODE steps** | $n_{\text{steps}}$ | 3–10 | Geodesic accuracy |
| **LBFGS iterations** | $n_{\text{iter}}$ | 50–200 | Optimization convergence |
| **Atlas iterations** | max_iter | 5–15 | Fréchet mean convergence |
| **Atlas step size** | $\alpha$ | 0.3–1.0 | Update aggressiveness |
| **PCA components** | $n$ | 5–20 | Variance captured |

**Practical recommendations for femur meshes (~18k points, ~100mm bbox):**
- $\sigma = 10$–$15$ mm
- $\lambda = 0.01$–$0.05$
- $n_{\text{steps}} = 3$–$5$
- $n_{\text{iter}} = 50$–$100$
- GPU acceleration strongly recommended

---

## 9. References

[1] Arnold, V.I. (1966). "Sur la géométrie différentielle des groupes de Lie de dimension infinie et ses applications à l'hydrodynamique des fluides parfaits." *Annales de l'Institut Fourier*, 16(1), 319–361.

[7] Chen, R.T.Q., et al. (2018). "Neural Ordinary Differential Equations." *NeurIPS*.

[10] Durrleman, S., et al. (2014). "Morphometry of anatomical shape complexes with dense deformations and sparse parameters." *NeuroImage*, 101, 35–49.

[12] Fréchet, M. (1948). "Les éléments aléatoires de nature quelconque dans un espace distancié." *Annales de l'Institut Henri Poincaré*, 10(4), 215–310.

[13] Grenander, U. (1993). *General Pattern Theory*. Oxford University Press.

[17] Glaunès, J. (2005). "Transport par difféomorphismes de points, de mesures et de courants pour la comparaison de formes et l'anatomie numérique." PhD thesis, Université Paris 13.

[18] Joshi, S., et al. (2004). "Unbiased diffeomorphic atlas construction for computational anatomy." *NeuroImage*, 23, S151–S160.

[21] Lee, J.M. (1997). *Riemannian Manifolds: An Introduction to Curvature*. Springer.

[24] Miller, M.I., Trouvé, A., & Younes, L. (2006). "Geodesic Shooting for Computational Anatomy." *Journal of Mathematical Imaging and Vision*, 24(2), 209–228.

[26] Nocedal, J. (1980). "Updating quasi-Newton matrices with limited storage." *Mathematics of Computation*, 35(151), 773–782.

[32] scikit-shapes documentation. https://scikit-shapes.github.io/

[35] Trouvé, A. (1998). "Diffeomorphisms groups and pattern matching in image analysis." *International Journal of Computer Vision*, 28(3), 213–221.

**Additional resource:**
- Feydy, J. (2016–2019). "Introduction à la Géométrie Riemannienne par l'étude des Espaces de Formes." Lecture notes, ENS Ulm. [Comprehensive treatment in French]

---

## Appendix: Code-Theory Correspondence

| Theory Concept | Code Location | Function/Class |
|----------------|---------------|----------------|
| Hamiltonian $H(\mathbf{p}, \mathbf{q})$ | `skshapes/.../extrinsic_deformation.py` | `ExtrinsicDeformation.H()` |
| Hamilton's equations | `skshapes/.../extrinsic_deformation.py` | `ExtrinsicDeformation.ode_func()` |
| Geodesic shooting (Exp) | `lddmm/registration.py` | `LDDMMRegistration.shoot()` |
| Registration (Log) | `lddmm/registration.py` | `LDDMMRegistration.compute_momentum()` |
| Atlas building | `lddmm/atlas.py` | `AtlasBuilder.build()` |
| Tangent PCA | `lddmm/tangent_pca.py` | `TangentPCA.fit()` |
| Shape synthesis | `lddmm/tangent_pca.py` | `TangentPCA.synthesize_shape()` |
| Shape projection | `lddmm/tangent_pca.py` | `TangentPCA.project()` |
