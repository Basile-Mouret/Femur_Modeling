# PCA-Based Synthetic Data Generation: Analysis and Recommendations

## Context

This document analyzes the viability of using PCA-generated synthetic femur shapes to augment the training dataset for neural network methods.

### Dataset Characteristics

| Property | Value |
|----------|-------|
| Training samples | 22 femurs |
| Validation samples | 2 femurs |
| Vertices per shape | 18,291 |
| Dimensions | $D = 54,873$ (3 × vertices) |
| PCA components | 21 (= N - 1) |
| Population | Male/female, all ages, no extreme pathologies |
| Alignment | Pre-aligned in common "World" referential |

---

## 1. Mathematical Framework

### 1.1 PCA Shape Model

Given $N = 22$ training shapes $\{\mathbf{s}_1, \ldots, \mathbf{s}_N\}$ in $\mathbb{R}^D$, PCA computes:

1. **Mean shape**: $\bar{\mathbf{s}} = \frac{1}{N} \sum_{i=1}^N \mathbf{s}_i$

2. **Principal components**: $\{\mathbf{v}_1, \ldots, \mathbf{v}_K\}$ orthonormal eigenvectors of the covariance matrix

3. **Variances**: $\{\lambda_1, \ldots, \lambda_K\}$ corresponding eigenvalues, ordered $\lambda_1 \geq \lambda_2 \geq \ldots$

### 1.2 Shape Generation

A new shape is synthesized as:

$$\mathbf{s}_{new} = \bar{\mathbf{s}} + \sum_{k=1}^{K} \alpha_k \sqrt{\lambda_k} \cdot \mathbf{v}_k$$

where $\alpha_k \sim \mathcal{N}(0, 1)$ are independent standard normal coefficients.

**Constraint**: Typically $|\alpha_k| \leq 3$ to stay within 99.7% of the learned distribution.

### 1.3 The Linear Subspace

**Key insight**: All generated shapes lie in a $(K = 21)$-dimensional linear subspace:

$$\mathcal{S}_{PCA} = \left\{ \bar{\mathbf{s}} + \sum_{k=1}^{21} \beta_k \mathbf{v}_k : \beta_k \in \mathbb{R} \right\} \subset \mathbb{R}^{54,873}$$

This is a **hyperplane** passing through the mean, spanned by the principal components.

---

## 2. Arguments For Using PCA Augmentation

### 2.1 Increased Sample Count

Neural networks benefit from more training examples. PCA enables generating unlimited shapes:

| Source | Count |
|--------|-------|
| Original training data | 22 |
| PCA-generated (theoretically) | ∞ |
| Practical augmentation | 100–10,000 |

### 2.2 Valid Shape Interpolation

Generated shapes are linear combinations of observed variation modes. Staying within $\pm 3\sigma$ ensures:
- No anatomically impossible configurations
- Smooth interpolation between training examples
- Coverage of intermediate shapes not explicitly observed

### 2.3 Controlled Diversity

Systematic sampling covers the learned distribution:

```python
# Uniform coverage of shape space
for alpha_1 in np.linspace(-2, 2, 10):
    for alpha_2 in np.linspace(-2, 2, 10):
        shape = mean + alpha_1 * sqrt(λ1) * v1 + alpha_2 * sqrt(λ2) * v2 + ...
```

### 2.4 Computational Efficiency

Generation is instant:
- One matrix multiplication: $O(D \times K)$
- No data collection, scanning, or registration required

### 2.5 Shape Prior for Neural Networks

Training on PCA shapes teaches the network:
- What a "valid femur" looks like
- The principal modes of variation
- Useful for autoencoder pretraining and regularization

---

## 3. Arguments Against (Critical Limitations)

### 3.1 No New Information Created

**Fundamental theorem**: The PCA subspace has dimension at most $\min(D, N-1) = 21$.

$$\text{rank}(\text{Cov}) \leq N - 1 = 21$$

**Consequence**: 
- 22 original shapes span a 21D subspace
- 10,000 PCA-generated shapes span the **same** 21D subspace
- No amount of PCA sampling creates information outside this subspace

**Geometric interpretation**:
```
Original data:    22 points in ℝ⁵⁴⁸⁷³
PCA subspace:     21-dimensional hyperplane through mean
Generated data:   More points, SAME hyperplane
True shape space: Unknown, possibly much higher dimensional
```

### 3.2 Linear Assumption is Biologically Wrong

Real anatomical variation is **nonlinear**:

| Variation Type | Linear Model | Reality |
|----------------|--------------|---------|
| Femoral neck angle | Linear scaling | Rotation (nonlinear) |
| Bone curvature | Additive | Multiplicative/curved |
| Size scaling | Isotropic | Allometric (nonlinear) |
| Joint surfaces | Flat interpolation | Curved manifold |

PCA linearizes the shape manifold around the mean:

$$\mathbf{s} \approx \bar{\mathbf{s}} + J \cdot \boldsymbol{\alpha}$$

where $J = [\sqrt{\lambda_1}\mathbf{v}_1, \ldots, \sqrt{\lambda_K}\mathbf{v}_K]$ is the Jacobian at the mean.

This is a **first-order Taylor approximation** — accurate near the mean, increasingly wrong for extreme shapes.

### 3.3 Gaussian Assumption May Be Violated

PCA assumes the data follows a **unimodal Gaussian distribution**:

$$\mathbf{s} \sim \mathcal{N}(\bar{\mathbf{s}}, \Sigma)$$

**Dataset reality**: Male/female, all ages

This suggests potential **subpopulations**:

| Factor | Possible Effect |
|--------|-----------------|
| Sex | Dimorphism in size, proportions, angles |
| Age | Growth patterns, bone density changes |
| Body size | Correlated scaling |

If the true distribution is **multimodal** (e.g., bimodal for sex):

$$p(\mathbf{s}) = \pi_1 \mathcal{N}(\boldsymbol{\mu}_1, \Sigma_1) + \pi_2 \mathcal{N}(\boldsymbol{\mu}_2, \Sigma_2)$$

Then:
- PCA computes a **single global mean** between modes
- Sampling uniformly in PCA space **over-represents** the between-mode region
- Generated shapes may be "averaged" hybrids that don't correspond to real anatomy

**Visual intuition**:
```
True distribution:     ●●●        ●●●     (two clusters: male/female)
                       mode1      mode2
                       
PCA Gaussian fit:         ●●●●●●●         (single Gaussian centered between)
                            mean
                            
Generated samples:     ●  ● ●●●●●●● ●  ●  (many samples in low-density region)
```

### 3.4 Neural Network Learns the PCA Manifold

If trained exclusively on PCA-generated data, the network learns:

$$f_{NN}: \mathbb{R}^D \to \mathcal{S}_{PCA}$$

It maps all inputs to the 21D PCA subspace.

**Problem**: Real test shapes have components **orthogonal** to this subspace.

Decompose any shape:
$$\mathbf{s} = \underbrace{\bar{\mathbf{s}} + \sum_{k=1}^{21} \alpha_k \sqrt{\lambda_k} \mathbf{v}_k}_{\mathbf{s}_\parallel \in \mathcal{S}_{PCA}} + \underbrace{\mathbf{e}}_{\mathbf{s}_\perp \perp \mathcal{S}_{PCA}}$$

- PCA-generated: $\mathbf{e} = \mathbf{0}$ by construction
- Real new shapes: $\mathbf{e} \neq \mathbf{0}$ (contains information orthogonal to training)

A network trained only on PCA data **cannot learn to handle** $\mathbf{e}$.

### 3.5 Reconstruction Error Lower Bound

The PCA model cannot reconstruct any shape better than its projection:

$$\|\mathbf{s} - \hat{\mathbf{s}}_{PCA}\|^2 = \|\mathbf{e}\|^2 = \sum_{k > K} \alpha_k^2 \lambda_k$$

For shapes outside the training distribution, this error can be significant.

---

## 4. Quantitative Assessment

### 4.1 Variance Explained

From our trained model:

| Components K | Variance Explained | Interpretation |
|--------------|-------------------|----------------|
| 7 | 90% | Major shape variations |
| 10 | 95% | Most clinically relevant variation |
| 17 | 99% | Nearly complete (within training data) |
| 21 | 100% | Full training subspace |

### 4.2 What 95% Variance Means

"95% variance explained" means:
- 95% of **observed** variation in the 22 training shapes
- NOT 95% of all possible femur variation
- Unknown variation exists orthogonal to the training subspace

### 4.3 Effective Dimensionality

With 22 samples, we can reliably estimate at most ~20 independent directions. The true dimensionality of femur shape space is unknown but likely higher.

---

## 5. Recommendations

### 5.1 Use Cases

| Application | Recommendation | Rationale |
|-------------|----------------|-----------|
| Autoencoder pretraining | ✅ Recommended | Helps learn major modes |
| Learning shape priors | ✅ Recommended | Network learns "what a femur looks like" |
| Replacing real data | ❌ Avoid | Network won't generalize |
| Combined with real data | ✅ Best approach | Augment, don't replace |
| Validation/testing | ❌ Never | Always use held-out real shapes |

### 5.2 Best Practices

```python
# GOOD: Mix real and synthetic data
training_data = (
    real_shapes +                    # All 22 original shapes
    pca_generated_shapes +           # N synthetic shapes from PCA
    noise_augmented_real_shapes      # Real shapes + small perturbations
)

# CRITICAL: Validate ONLY on real held-out data
validation_data = real_validation_shapes  # The 2 validation femurs

# NEVER do this:
validation_data = pca_generated_shapes  # WRONG - circular validation
```

### 5.3 Additional Augmentation Strategies

To complement PCA generation:

| Method | Description | Benefit |
|--------|-------------|---------|
| **Gaussian noise** | Add $\epsilon \sim \mathcal{N}(0, \sigma^2 I)$ to vertices | Robustness to measurement noise |
| **Rigid transforms** | Random rotation/translation | Invariance (though data is pre-aligned) |
| **Elastic deformation** | Small smooth deformations | Local variation not in PCA |
| **Dropout augmentation** | Randomly mask vertices | Robustness to missing data |
| **Mixup** | Interpolate between real shapes | Smooth decision boundaries |

### 5.4 Sampling Strategy

Given the heterogeneous population (male/female, all ages), consider:

1. **Stratified sampling**: If sex/age labels available, sample PCA coefficients separately per group
2. **Truncated sampling**: Avoid extreme $|\alpha_k| > 2.5$ which may produce implausible shapes
3. **Density-aware sampling**: Weight samples by estimated density to avoid oversampling low-density regions

---

## 6. Conclusion

### 6.1 Will It Work?

**For autoencoder/VAE training**: **Yes, with caveats**
- The network will learn a (possibly better) nonlinear representation
- Useful for compression, denoising, interpolation within the training distribution
- Limited generalization to truly novel anatomy

**For general neural network training**: **Partially**
- Improves data efficiency and regularization
- Does not replace the need for more real data
- Cannot extrapolate beyond the 21D learned subspace

### 6.2 Key Takeaway

$$\boxed{\text{PCA augmentation is a regularizer, not a data source}}$$

It biases the network toward the learned shape space. This is beneficial if the 22 training femurs adequately represent the target population, but cannot compensate for fundamentally insufficient sampling of human anatomical variability.

### 6.3 Future Directions

To overcome PCA limitations:
1. **Collect more real data** — most effective but expensive
2. **Use VAE** — learns nonlinear manifold, better generative model
3. **Use LDDMM** — geodesic interpolation respects anatomical constraints
4. **Domain adaptation** — transfer from larger public datasets

---

*Document created: January 2026*  
*Project: Femur Shape Modeling - Statistical Shape Analysis*
