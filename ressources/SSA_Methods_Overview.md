# Statistical Shape Analysis Methods for Femur Modeling

## Overview

This document provides a comprehensive review of methods for **Statistical Shape Analysis (SSA)** applied to femur bone data. The goal is to understand different mathematical approaches for shape modeling, enabling tasks such as:

- **Mean shape computation**
- **Variance analysis** (understanding how shapes vary in a population)
- **Dimensionality reduction** (compact representation of high-dimensional shape data)
- **Shape reconstruction** (from partial data or constraints)
- **Shape synthesis** (generating realistic new shapes)
- **Atlas building** (creating a representative template for a population)

We organize the methods into three "levels" of increasing mathematical sophistication:

1. **Level 1: Linear PCA** - Classical statistical approach
2. **Level 2: Neural Networks** - Deep learning for nonlinear representation
3. **Level 3: LDDMM** - Diffeomorphic methods based on Riemannian geometry

---

## 1. Linear PCA-Based Statistical Shape Model (SSM)

### 1.1 Theoretical Foundation

**Principal Component Analysis (PCA)** is the classical method for linear dimensionality reduction in shape spaces.

#### Key Concepts

- **Shape Representation**: Each femur is represented as a vector in $\mathbb{R}^D$ where $D = 3N$ for $N$ vertices (each with 3D coordinates). For the project dataset: $D = 3 \times 18,291 = 54,873$ dimensions.

- **Correspondence**: All shapes must have **point correspondence** - vertex $i$ in all shapes must represent the same anatomical location. **This is provided in the dataset** (meshes have been pre-registered with consistent vertex ordering).

- **Covariance Matrix**: Given $M$ samples, compute the sample mean $\bar{\mathbf{F}}$ and the $D \times D$ covariance matrix:
$$\text{Cov}(\mathbf{F}', \mathbf{F}') = \frac{1}{M-1} \sum_{i=1}^{M} (\mathbf{F}_i - \bar{\mathbf{F}})(\mathbf{F}_i - \bar{\mathbf{F}})^T$$

- **Eigendecomposition**: Diagonalize the covariance matrix to find principal components (eigenvectors) and variances (eigenvalues):
$$\text{Cov} = R \Lambda R^T$$

#### The SSM Model

A new shape can be synthesized as:
$$\mathbf{F} = \bar{\mathbf{F}} + \sum_{k=1}^{K} \alpha_k \sqrt{\lambda_k} \mathbf{v}_k$$

Where:
- $\bar{\mathbf{F}}$ is the mean shape
- $\mathbf{v}_k$ are the principal modes (eigenvectors)
- $\lambda_k$ are the corresponding eigenvalues (variances)
- $\alpha_k \sim \mathcal{N}(0, 1)$ are the mode weights

### 1.2 Practical Implementation (SVD)

Since $D >> M$ (54,873 dimensions but only 24 samples), use **Singular Value Decomposition (SVD)** on the data matrix instead of eigendecomposition of the huge covariance matrix:

```
D = U Σ V^T
```

Where:
- Eigenvalues = (singular values)²
- The rank of the covariance is at most $M-1 = 23$

### 1.3 Goals Achievable with Linear PCA

| Goal | Method |
|------|--------|
| **Mean Shape** | Compute $\bar{\mathbf{F}} = \frac{1}{M}\sum_i \mathbf{F}_i$ |
| **Variance Analysis** | Examine eigenvalues $\lambda_k$ and visualize deformation modes |
| **Dimensionality Reduction** | Keep only top $K$ modes (e.g., capturing 95% of variance) |
| **Shape Reconstruction** | Project partial/noisy data onto principal subspace |
| **Shape Synthesis** | Sample $\alpha_k$ from Gaussian, generate new shapes |

### 1.4 Limitations

- **Linearity assumption**: Real anatomical variations may be nonlinear
- **Gaussian assumption**: Shapes may not follow a Gaussian distribution
- **Requires correspondence**: Vertices must be semantically matched across shapes
- **Limited modes**: With $M$ samples, at most $M-1$ modes are available

### 1.5 References

- Cootes, T.F., Taylor, C.J., et al. (1995). "Active Shape Models—Their Training and Application." *Computer Vision and Image Understanding*.
- Project slides: `ressources/subject_femur_modeling.pdf`

---

## 2. Kernel PCA (kPCA) - Nonlinear Extension

### 2.1 Theoretical Foundation

**Kernel PCA** extends PCA to capture nonlinear relationships by implicitly mapping data to a higher-dimensional feature space.

#### The Kernel Trick

Instead of computing in the explicit feature space $\Phi(\mathbf{x})$, we work with a **kernel function**:
$$K(\mathbf{x}, \mathbf{y}) = \Phi(\mathbf{x})^T \Phi(\mathbf{y})$$

Common kernels:
- **Polynomial**: $K(\mathbf{x}, \mathbf{y}) = (\mathbf{x}^T \mathbf{y} + c)^d$
- **Gaussian (RBF)**: $K(\mathbf{x}, \mathbf{y}) = \exp\left(-\frac{\|\mathbf{x} - \mathbf{y}\|^2}{2\sigma^2}\right)$

#### Algorithm

1. Compute the $M \times M$ kernel matrix $K_{ij} = K(\mathbf{F}_i, \mathbf{F}_j)$
2. Center the kernel matrix: $K' = K - \frac{1}{M}K\mathbf{1} - \mathbf{1}K/M + \mathbf{1}K\mathbf{1}/M^2$
3. Solve the eigenvalue problem: $M\lambda \mathbf{a} = K' \mathbf{a}$
4. Project new data: $\text{proj}_k(\mathbf{F}) = \sum_{i=1}^{M} a_i^k K(\mathbf{F}_i, \mathbf{F})$

### 2.2 Goals Achievable

- **Nonlinear dimensionality reduction**
- **Better separation of shape clusters**
- **Capture nonlinear deformation patterns**

### 2.3 Limitations

- **Pre-image problem**: Cannot easily reconstruct shapes from kernel space
- **Kernel selection**: Choice of kernel and parameters is critical
- **Scalability**: Kernel matrix is $M \times M$

### 2.4 References

- Schölkopf, B., Smola, A., Müller, K.R. (1998). "Nonlinear Component Analysis as a Kernel Eigenvalue Problem." *Neural Computation*.
- YouTube resources in project slides

---

## 3. Autoencoders for Shape Analysis

### 3.1 Basic Autoencoder

An autoencoder learns a nonlinear mapping from high-dimensional shape space to a low-dimensional **latent space** and back.

#### Architecture

```
         Encoder                    Decoder
Shape ─────────────> Latent Code ─────────────> Reconstructed Shape
(D dim)              (k << D dim)               (D dim)
```

**Loss function**:
$$L(\theta, \phi) = \frac{1}{M} \sum_{i=1}^{M} \|\mathbf{F}_i - D_\theta(E_\phi(\mathbf{F}_i))\|^2$$

#### Relationship to PCA

- **Linear autoencoder** (one hidden layer, linear activations) learns the same subspace as PCA
- **Nonlinear autoencoder** (multiple layers, nonlinear activations) can capture nonlinear manifold structure

### 3.2 Variational Autoencoder (VAE)

VAEs add a **probabilistic** framework that enables:
- Principled sampling from latent space
- Regularized, smooth latent representations

#### ELBO Loss

$$\mathcal{L}_{\theta,\phi}(\mathbf{x}) = \mathbb{E}_{z \sim q_\phi(z|\mathbf{x})}[\log p_\theta(\mathbf{x}|z)] - D_{KL}(q_\phi(z|\mathbf{x}) \| p(z))$$

- **Reconstruction term**: Decoder should reconstruct input well
- **KL divergence term**: Encoder distribution should be close to prior (usually standard Gaussian)

#### Reparameterization Trick

To enable backpropagation through sampling:
$$z = \mu_\phi(\mathbf{x}) + \sigma_\phi(\mathbf{x}) \odot \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)$$

### 3.3 Goals Achievable with Autoencoders

| Goal | Method |
|------|--------|
| **Dimensionality Reduction** | Bottleneck layer (latent code) |
| **Mean/Variance Analysis** | Analyze latent space distribution |
| **Shape Synthesis** | Sample from latent space, decode |
| **Reconstruction** | Encode partial data, decode full shape |
| **Interpolation** | Linear interpolation in latent space |

### 3.4 Advantages Over PCA

- **Nonlinear** representation learning
- **More compact** latent codes (e.g., 64 dimensions vs 23 for PCA)
- Can handle **missing data** better
- **Generative** capability (especially VAE)

### 3.5 Implementation in C++

The project already has a neural network implementation (`neuralNetwork.hpp`). To create an autoencoder:

1. Use symmetric architecture: e.g., `[54873, 1024, 256, 64, 256, 1024, 54873]`
2. Train with MSE loss between input and output
3. Extract the middle layer as latent representation

### 3.6 References

- Kingma, D.P., Welling, M. (2014). "Auto-Encoding Variational Bayes." *ICLR*.
- Hinton, G.E., Salakhutdinov, R.R. (2006). "Reducing the Dimensionality of Data with Neural Networks." *Science*.

---

## 4. LDDMM - Large Deformation Diffeomorphic Metric Mapping

### 4.1 Theoretical Foundation

**LDDMM** is a powerful framework from **computational anatomy** that models shape deformations as **geodesics in the space of diffeomorphisms** (smooth, invertible transformations).

#### Key Concepts

1. **Shape Space as a Manifold**: Shapes are viewed as points on an infinite-dimensional curved manifold

2. **Diffeomorphisms**: Smooth, invertible mappings $\phi: \mathbb{R}^d \to \mathbb{R}^d$ that transform one shape into another

3. **Flows of Velocity Fields**: Diffeomorphisms are generated by integrating time-dependent velocity fields:
$$\frac{\partial \phi_t(x)}{\partial t} = v_t(\phi_t(x)), \quad \phi_0(x) = x$$

4. **Riemannian Metric**: A right-invariant metric on the group of diffeomorphisms:
$$d_G(\text{id}, \phi) = \inf \left\{ \int_0^1 \|v_t\|_V dt : \phi_1^v = \phi \right\}$$

5. **RKHS (Reproducing Kernel Hilbert Space)**: The space $V$ of velocity fields is defined using a kernel $K_V$:
$$\|v\|_V^2 = \langle v, v \rangle_V$$

#### Geodesic Shooting

Instead of optimizing the full path $v_t$, we can parameterize by **initial momentum** $m_0$:
$$v_t(x) = \sum_{i=1}^n K_V(x, q_i(t)) p_i(t)$$

The geodesic equations in Hamiltonian form:
$$\begin{cases}
\dot{q} = K_V(q,q) p \\
\dot{p} = -\frac{1}{2} \nabla_q \langle K_V(q,q) p, p \rangle
\end{cases}$$

### 4.2 Matching Problem

To match source shape $S$ to target shape $T$:
$$J(\phi) = \gamma \cdot d_G(\text{id}, \phi)^2 + A(\phi \cdot S, T)$$

Where:
- $d_G^2$ is the squared geodesic distance (regularization)
- $A$ is a data attachment term (e.g., sum of squared distances between corresponding points)
- $\gamma$ balances regularity vs fidelity

### 4.3 Atlas Building

Given a population of shapes $\{S_1, \ldots, S_M\}$, find an atlas $\hat{I}$ and deformations $\{\phi_i\}$ such that:
$$\hat{I} = \arg\min_I \sum_i \left[ \min_{m_i} \|\phi_i \cdot I - S_i\|^2 + \lambda \|m_i\|_{K_V}^2 \right]$$

### 4.4 Diffeomorphic Autoencoders

A recent advancement combines LDDMM with deep learning:

**Architecture**:
```
Image/Shape ──[Encoder NN]──> Initial Momentum ──[EPDiff Integration]──> Deformation ──[Apply to Atlas]──> Reconstruction
```

**Benefits**:
- Deep network learns to predict optimal momentum directly
- Atlas and network trained jointly
- Enables minibatch SGD (efficient for large datasets)
- Deformations are guaranteed diffeomorphic (smooth, invertible)
- Latent space interpolation produces anatomically plausible intermediate shapes

### 4.5 Goals Achievable with LDDMM

| Goal | Method |
|------|--------|
| **Mean Shape (Atlas)** | Fréchet mean in diffeomorphism group |
| **Variance Analysis** | PCA on initial momenta (tangent space) |
| **Dimensionality Reduction** | Low-dim representation of momenta |
| **Registration** | Optimal diffeomorphism between shapes |
| **Shape Synthesis** | Sample momenta, shoot geodesics from atlas |
| **Interpolation** | Geodesic paths between shapes |

### 4.6 Advantages

- **Topology preservation**: Diffeomorphisms are invertible, no folding
- **Anatomically meaningful**: Deformations respect spatial structure
- **Riemannian statistics**: Proper geometric framework for shape statistics
- **Point correspondence not required** (with varifold/current representations)

### 4.7 Challenges

- **Computationally expensive**: Integration of PDEs
- **Complex implementation**: Requires careful numerics
- **Many hyperparameters**: Kernel scale, regularization, etc.

### 4.8 References

- Beg, M.F., Miller, M.I., Trouvé, A., Younes, L. (2005). "Computing Large Deformation Metric Mappings via Geodesic Flows of Diffeomorphisms." *IJCV*.
- Hinkle, J., Womble, D., Yoon, H.-J. (2019). "Diffeomorphic Autoencoders for LDDMM Atlas Building." *MIDL*.
- Glaunès, J.A. "The Diffeomorphic Framework for Shape Analysis" (summer school slides)
- Project resources: `ressources/LDDMM/`

---

## 5. Comparison of Methods

| Aspect | Linear PCA | Kernel PCA | Autoencoders | LDDMM |
|--------|------------|------------|--------------|-------|
| **Linearity** | Linear | Nonlinear (implicit) | Nonlinear | Nonlinear |
| **Correspondence Required** | Yes | Yes | Yes | Optional |
| **Computational Cost** | Low | Medium | Medium-High | High |
| **Interpretability** | High (modes) | Low | Medium | High (geodesics) |
| **Generative** | Yes | Difficult | Yes (VAE) | Yes |
| **Topology Preservation** | No guarantee | No guarantee | No guarantee | Guaranteed |
| **Implementation** | Easy | Medium | Medium | Hard |

---

## 6. Practical Recommendations for This Project

### Phase 1: Linear PCA (Foundation)
1. Load femur data (already implemented in `Femur` class)
2. Stack vertex coordinates into shape vectors
3. Compute mean shape and SVD
4. Visualize principal modes of deformation
5. Implement shape synthesis from PCA model

### Phase 2: Autoencoder (Neural Network Extension)
1. Use existing `NeuralNetwork` class to build autoencoder
2. Define symmetric architecture
3. Train on femur dataset
4. Compare latent space dimensionality with PCA
5. Evaluate reconstruction quality

### Phase 3: Advanced Methods (Optional/Research)
1. **Kernel PCA**: Implement with RBF kernel
2. **VAE**: Extend autoencoder with KL divergence loss
3. **LDDMM**: Would require significant new development
   - Consider Python implementation (PyTorch + lagomorph library)
   - C++ bindings for integration with existing code

### Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                        Python Layer                          │
│  • Visualization (viewer3D.py)                               │
│  • High-level analysis scripts                               │
│  • LDDMM (using lagomorph/PyTorch)                          │
├─────────────────────────────────────────────────────────────┤
│                        C++ Core Library                       │
│  • Femur data loading/saving (femur.cpp)                     │
│  • Linear algebra (linalg.cpp) - Eigen backend               │
│  • PCA computation                                           │
│  • Neural networks (neuralNetwork.cpp)                       │
│  • Autoencoders                                              │
├─────────────────────────────────────────────────────────────┤
│                        Data Layer                             │
│  • OBJ files (training/validation)                           │
│  • Saved models (binary format)                              │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Mathematical Summary

### Shape Space Notation

- Shape: $\mathbf{F} \in \mathbb{R}^{3N}$ (flattened vertex coordinates)
- Population: $\{\mathbf{F}_1, \ldots, \mathbf{F}_M\}$
- Mean: $\bar{\mathbf{F}} = \frac{1}{M} \sum_i \mathbf{F}_i$
- Covariance: $C = \frac{1}{M-1} \sum_i (\mathbf{F}_i - \bar{\mathbf{F}})(\mathbf{F}_i - \bar{\mathbf{F}})^T$

### PCA Model

$$\mathbf{F} \approx \bar{\mathbf{F}} + \Phi_K \mathbf{b}$$

where $\Phi_K = [\phi_1 | \cdots | \phi_K]$ are the first $K$ eigenvectors and $\mathbf{b} \in \mathbb{R}^K$ are shape coefficients.

### Autoencoder

$$\mathbf{F}' = D_\theta(E_\phi(\mathbf{F}))$$

where $E_\phi: \mathbb{R}^D \to \mathbb{R}^k$ (encoder) and $D_\theta: \mathbb{R}^k \to \mathbb{R}^D$ (decoder).

### LDDMM

$$\phi_1 = \text{id} + \int_0^1 v_t \circ \phi_t \, dt$$

$$v_t = K_V * (D\phi_t^{-T} \cdot m_t \circ \phi_t^{-1})$$

---

## 8. Further Reading

### Books
- Dryden, I.L., Mardia, K.V. (2016). *Statistical Shape Analysis*. Wiley.
- Younes, L. (2010). *Shapes and Diffeomorphisms*. Springer.
- Goodfellow, I., Bengio, Y., Courville, A. (2016). *Deep Learning*. MIT Press.

### Online Resources
- Deep Representation Learning Book: https://ma-lab-berkeley.github.io/deep-representation-learning-book/
- Kernel PCA tutorial (YouTube): https://www.youtube.com/watch?v=HbDHohXPLnU
- Autoencoders tutorial: https://www.youtube.com/watch?v=MnRskV3NY1k
- Lagomorph library (LDDMM in PyTorch): https://github.com/jacobhinkle/lagomorph

### Key Papers
1. Cootes et al. (1995) - Active Shape Models
2. Schölkopf et al. (1998) - Kernel PCA
3. Kingma & Welling (2014) - Variational Autoencoders
4. Beg et al. (2005) - LDDMM
5. Hinkle et al. (2019) - Diffeomorphic Autoencoders

---

## Appendix A: Existing Code Structure

### Current C++ Implementation

| File | Purpose |
|------|---------|
| `femur.hpp/cpp` | Femur mesh data structure, OBJ I/O |
| `linalg.hpp/cpp` | Matrix/Vector classes (Eigen backend) |
| `neuralNetwork.hpp/cpp` | MLP implementation |
| `neuralNetworkFunctions.hpp/cpp` | Activation/loss functions |

### Extensions Needed

| Component | For Method | Priority |
|-----------|------------|----------|
| SVD computation | PCA | High |
| Shape vector packing/unpacking | All | High |
| PCA class | PCA | High |
| Autoencoder architecture | AE/VAE | Medium |
| KL divergence loss | VAE | Medium |
| Gaussian kernel | kPCA | Low |
| LDDMM geodesic shooting | LDDMM | Research |

---

*Document created: January 2026*
*Project: Femur Shape Modeling - Statistical Shape Analysis*
