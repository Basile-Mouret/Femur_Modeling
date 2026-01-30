// LDDMM Theory and Implementation
// This chapter explains the mathematical foundations of our LDDMM implementation

#figure(
  caption: []
)[
  #image("/resources/img/fishdarcy.png", width: 80%) 
]
In an altogether remarkable work, the scholar, biologist, and mathematician D’Arcy Wentworth Thompson (1860-1948) underscored the importance of environmental and physical factors (as opposed to heredity alone) in the morphogenesis of living beings. Since the shape of fish is more or less optimal, there is not an infinite number of distinct "plans," but rather a mere handful of original patterns which allow, through (non-trivial) deformations, the generation of all forms favored by evolution.
    
    To describe the anatomical variability of an observed family or population, it is therefore sufficient to provide a _reference template_ (arbitrarily complex, yet common to all observations) and the deformations required to map said template to the individuals. Complexity is thus decoupled into two intelligible components: a _reference image_, complex but fixed; and _deformations_ specific to the observed subjects, often simple enough to be described with few parameters.
    
    Remarkable fact: the diagrams above present the variability of fish shapes not as arbitrary displacements of skeletons, but as *coordinate changes, deformations of the ambient space*. It is upon this mathematical concept of _extrinsic_ deformation of space (as opposed to _intrinsic_ movements of fish particles) that Procrustean analysis and the “LDDMM” theory presented in this chapter rely.

    We will attempt to present the big ideas and algorithms of LDDMM for shape spaces with point correspondance (commonly known as landmarks spaces) in an intuitive manner, preserving the most mathematical content as possible but surely making compromises. For full technical details, proofs and rigorous presentation, see @younes2010shapes, @beg2005computing , @miller2015hamiltonian, @joshi2000landmark . 
    This study will be centered on the application of LDDMM to human femurs, but the global theory is a general computational anatomy framework.

== Overview
*Large Deformation Diffeomorphic Metric Mapping* (LDDMM) is a mathematical framework for computing smooth, invertible transformations between shapes @beg2005computing. Unlike linear methods that treat shapes as vectors in Euclidean space, LDDMM treats shapes as points on a *Riemannian manifold*, i.e a curved locally Euclidean space, respecting the intrinsic geometry of the shape space.
Consider the space of all possible femur shapes $cal(S)$. There is no biological nor anatomical reason suggesting this should be a vector space---you cannot simply add two femurs and get a valid femur [SOURCE]. Studying large deformations with a Riemannian approach has been an efficient point of view to generate metrics between deformable objects, and to provide accurate, non ambiguous and smooth matchings between shapes. 

#figure(
  caption: [Curved space is not stable by linear operations, and requires a  metric aware of its geometry]
)[
  #image("/resources/img/sphere_geodesic.png", width: 80%) 
]
In addition to anatomical implausibility of linear transformations, the Euclidean distance poses other challenges to anatomical relevancy, as it treats all point movements equally : this is not the general case in nature, as some deformations physically "cost" more than others, or simply less probable. The straight line $S_1 -> S_2$ is not the optimal deformation path on the shape manifold.

For statistical shape analysis, we need two fundamental objects:
- A *distance* between shapes (how different are two femurs?)
- A *linear space* for statistics (PCA requires vector operations)


LDDMM provides both through its Riemannian geometry. Crucially, the resulting mean and distances are geometrically meaningful and respect the physical constraints of anatomical deformations @younes2010shapes. We first present the broad theoretical plan, before diving into technical aspects of the construction aimed at making the problem computationally tractable.

Our goal is to build/learn a relevant metric on $cal(D) = op("Diff")(cal(S))$, the group of diffeomorphisms (smooth invertible transformations) of $cal(S)$. The "cost" of these deformations will be evalued by the *geodesic distance* $d(S_1,S_2)$ on $cal(S)$ induced by this metric. It is computed as the *energy* of the cheapest transformation from $S_1$ to $S_2$. The notion of geodesic is a generalization of the notion of a "straight line" where every step of the movement must lie on the manifold. Using this distance, we can define a mean shape of our population of $K$ shapes, commonly referred to as *atlas* in Riemannian geometry.

$ macron(S) = arg min_S sum_(i=1)^K d^2 (S, S_i) $

The geodesic point of view of deformations is not only good for defining a distance.
Since $op("Diff")(cal(S))$ is curved and infinite dimensional, we can not perform standard statistical analysis on it. We thus restrict our study to geodesic deformations for three compelling reasons :

- Since our space of deformations is a manifold, it is locally Euclidean, i.e a flat vector space, called the *tangent space*. "Small" deformations from the mean, which we're interested in, thus live in $cal(T)_D macron(S)$, the tangent space at the atlas.
- A fundamental theorem on the nature of geodesics creates a one-to-one correspondance between geodesics from a point and 
=== The Tangent Space

At any shape (point on the manifold), there exists a *tangent space*: a linear vector space where we can perform standard operations like addition, subtraction, and inner products.

The *initial momentum* $bold(p)_0$ at the atlas encodes "which direction and how far" to travel along a geodesic to reach a target shape. Critically, the momentum $bold(p)_0$ is *not* the displacement $("target" - "source")$. It is the initial velocity of the geodesic, transformed through the kernel.

=== The Key Idea

Instead of directly computing a deformation $phi: RR^3 -> RR^3$, LDDMM constructs $phi$ as the *flow of a time-varying velocity field* $v(x, t)$:

$ (diff phi_t) / (diff t) (x) = v(phi_t (x), t), quad phi_0 = "Id" $

This construction ensures two fundamental properties:
+ *Diffeomorphism*: The transformation $phi$ is smooth and invertible---no folding or tearing can occur.
+ *Metric structure*: The "length" of the deformation path defines a proper distance between shapes.

=== Fundamental Difference from Euclidean Methods

LDDMM treats shapes as manifold-valued data, which leads to fundamentally different computations than Euclidean approaches:

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*Aspect*], [*Euclidean (Linear PCA)*], [*LDDMM*]),
    [Shape space], [$RR^(3N)$ (flat vector space)], [$"Diff"(Omega)$ (curved manifold)],
    [Distance], [$norm(S_1 - S_2)$ (Euclidean)], [$integral_0^1 norm(v_t)_V^2 dif t$ (geodesic)],
    [Mean], [Arithmetic: $1/K sum_i S_i$], [Fréchet: $arg min sum_i d^2 (mu, S_i)$],
    [Interpolation], [Linear: $(1-t)S_1 + t S_2$], [Geodesic: $"Exp"_(S_1)(t dot "Log"_(S_1)(S_2))$],
  ),
  caption: [Comparison between Euclidean and LDDMM approaches to shape analysis.]
) <tab:lddmm-comparison>

The distance in LDDMM is *not* the Euclidean distance, even when shapes have point correspondence. The geodesic distance measures the "cost" of the smoothest diffeomorphism connecting two shapes, weighted by a kernel that enforces spatial coherence.

=== Why Does This Matter?

For statistical shape analysis, we need three fundamental operations:
- A *distance* between shapes (how different are two femurs?)
- A *mean* shape (what is the average femur?)
- A *linear space* for statistics (PCA requires vector operations)

LDDMM provides all three through its Riemannian geometry. Crucially, the resulting mean and distances are geometrically meaningful and respect the physical constraints of anatomical deformations @younes2010shapes.

== Geodesic Shooting

=== The EPDiff Equation

LDDMM computes geodesics (shortest paths) on the shape manifold. The velocity field $v(t)$ evolves according to the *EPDiff equation* (Euler-Poincaré equation on diffeomorphisms) @miller2015hamiltonian:

$ (diff bold(p)) / (diff t) + nabla_v bold(p) + (nabla v)^T bold(p) + bold(p) thin "div"(v) = 0 $

where $bold(p)$ is the momentum, dual to velocity via the kernel: $v = K * bold(p)$.

=== The Exponential Map

Given an initial momentum $bold(p)_0$ at the atlas, we can compute the resulting shape by integrating the geodesic equations forward in time. This operation is called the *exponential map*:

$ "Exp"_"atlas" (bold(p)_0) = phi_1 ("atlas") $

The exponential map "shoots" from the atlas along the initial momentum direction to produce a new shape at time $t = 1$.

=== The Log Map

The inverse operation---finding the initial momentum that reaches a given target shape---is the *log map*:

$ "Log"_"atlas" (S) = bold(p)_0 quad "such that" quad "Exp"_"atlas" (bold(p)_0) = S $

Computing the log map requires solving an optimization problem (registration). It is *not* simply the difference $(S - "atlas")$.

=== The Geodesic Distance

The distance between two shapes is defined as the length of the geodesic connecting them:

$ d(S_1, S_2)^2 = min_({v_t}) integral_0^1 norm(v_t)_V^2 dif t $

where $v_t$ is the time-varying velocity field and $norm(dot)_V$ is the RKHS norm defined by the kernel. This geodesic distance is fundamentally different from the Euclidean distance.

== Energy Minimization and Registration

Computing the geodesic distance---or equivalently, the log map---requires solving an optimization problem. This section details the exact energy formulation used in our implementation via scikit-shapes @skshapes.

=== The Total Energy Functional

Given a source shape $S_"source"$ with vertices $bold(q)_0 in RR^(N times 3)$ and a target shape $S_"target"$, LDDMM registration finds the initial momentum $bold(p)_0 in RR^(N times 3)$ by minimizing:

$ E(bold(p)_0) = underbrace(cal(L)_"fid" (phi_1 (S_"source"), S_"target"), "Fidelity term") + lambda dot underbrace(H(bold(p)_0, bold(q)_0), "Regularization") $

where:
- $phi_1$ is the diffeomorphism at time $t=1$ generated by shooting from $bold(p)_0$
- $cal(L)_"fid"$ is the fidelity loss measuring correspondence quality
- $H$ is the Hamiltonian (kinetic energy of the deformation)
- $lambda$ is the regularization weight balancing data fit vs. smoothness

=== The Fidelity Term

For shapes with known point correspondence (our case), we use the *L2 loss*:

$ cal(L)_"fid" (phi_1 (S), T) = sum_(i=1)^N norm(phi_1 (bold(q)_i) - bold(t)_i)^2 $

where $bold(q)_i$ are source vertices and $bold(t)_i$ are corresponding target vertices.

=== The Regularization Term: Hamiltonian Formulation

The regularization is the *Hamiltonian* (kinetic energy) of the initial state:

$ H(bold(p), bold(q)) = 1/2 angle.l bold(p), K_(bold(q)) bold(p) angle.r = 1/2 sum_(i,j=1)^N bold(p)_i^T K(bold(q)_i, bold(q)_j) bold(p)_j $

where $K(bold(q)_i, bold(q)_j)$ is the kernel matrix evaluated between vertices. With a Gaussian kernel:

$ K(bold(q)_i, bold(q)_j) = exp(-norm(bold(q)_i - bold(q)_j)^2 / (2 sigma^2)) $

This quadratic form penalizes non-smooth deformations:
- Large momenta at isolated points lead to high energy (discouraged)
- Spatially coherent momenta lead to lower energy (encouraged)

The kernel scale $sigma$ determines the correlation length: nearby points (within distance $sigma$) are encouraged to move together.

=== Hamiltonian Dynamics: The Geodesic Equations

Given initial momentum $bold(p)_0$, the shape and momentum evolve according to *Hamilton's equations*:

$ dot(bold(q))_i = (diff H) / (diff bold(p)_i) = sum_(j=1)^N K(bold(q)_i, bold(q)_j) bold(p)_j $

$ dot(bold(p))_i = -(diff H) / (diff bold(q)_i) $

These ODEs are integrated from $t=0$ to $t=1$ using numerical solvers. The final positions $bold(q)_1 = phi_1 (bold(q)_0)$ give the morphed shape.

A crucial property of Hamiltonian systems is *energy conservation*:

$ H(bold(p)_t, bold(q)_t) = H(bold(p)_0, bold(q)_0) quad forall t in [0, 1] $

=== Connection to Geodesic Distance

For velocity fields parameterized by momenta, the RKHS norm becomes:

$ norm(v_t)_V^2 = 2 H(bold(p)_t, bold(q)_t) $

By energy conservation along geodesics:

$ d(S_1, S_2)^2 = integral_0^1 2 H(bold(p)_t, bold(q)_t) dif t = 2 H(bold(p)_0, bold(q)_0) $

Thus, *the squared geodesic distance equals twice the initial Hamiltonian* of the optimal momentum. This is why minimizing the total energy $E(bold(p)_0) = cal(L)_"fid" + lambda H$ simultaneously:
+ Achieves correspondence (via fidelity term)
+ Finds the shortest geodesic (via Hamiltonian regularization)

=== The Optimization Algorithm

We use *L-BFGS* (Limited-memory Broyden–Fletcher–Goldfarb–Shanno), a quasi-Newton optimization method that:
+ Approximates the Hessian using gradient history (memory-efficient)
+ Converges superlinearly near the optimum
+ Is well-suited for smooth, high-dimensional problems like LDDMM

The gradient $nabla_(bold(p)_0) E$ is computed via automatic differentiation through the ODE integration, leveraging PyTorch's `autograd` capabilities.

== The Kernel and RKHS

=== Why Regularization?

Without constraints, the velocity field $v$ could be arbitrarily irregular. We require *smooth* deformations that preserve anatomical plausibility---a heart should deform smoothly, not with discontinuous jumps.

=== The RKHS Framework

We constrain $v$ to live in a *Reproducing Kernel Hilbert Space (RKHS)* $V$ defined by a kernel $K$ @younes2010shapes:

$ norm(v)_V^2 = angle.l v, v angle.r_V $

The kernel implicitly defines the smoothness properties of allowed velocity fields.

=== The Gaussian Kernel

We use a Gaussian kernel:

$ K(x, y) = exp(-norm(x - y)^2 / (2 sigma^2)) $

*The scale parameter $sigma$ controls correlation:*
- *Small $sigma$*: Local deformations, each point moves relatively independently
- *Large $sigma$*: Global deformations, nearby points are strongly coupled

=== Choosing $sigma$ for Femurs

Rule of thumb: $sigma approx 10"-"20%$ of the shape's bounding box diagonal.

For femurs with approximately 100mm bounding box: $sigma approx 10"-"15"mm"$.

== Atlas Building

=== The Fréchet Mean

The atlas $mu$ is the *Fréchet mean*: the shape that minimizes total squared geodesic distance to all training shapes @durrleman2014morphometry:

$ mu = arg min_S sum_(i=1)^K d^2 (S, S_i) $

where $d$ is the *geodesic distance*, not the Euclidean distance.

=== Why the Fréchet Mean $eq.not$ Arithmetic Mean

Even with point correspondence, the Fréchet mean in LDDMM is *not* the arithmetic mean. Here's why:

+ *Different distance metric*: The geodesic distance $d(S_1, S_2)$ involves minimizing $integral norm(v_t)_V^2 dif t$ over smooth velocity fields. This is not equal to $norm(S_1 - S_2)_F$.

+ *Regularization matters*: The kernel $K$ penalizes non-smooth deformations. Two shapes differing by a smooth global rotation have smaller geodesic distance than shapes differing by local jagged displacements of the same Euclidean magnitude.

+ *Curved geometry*: The minimizer of $sum_i d^2 (mu, S_i)$ on a curved manifold is generally not the same as $(1/K) sum_i S_i$.

=== Algorithm: Iterative Geodesic Averaging

Our implementation uses true LDDMM geodesic averaging:

+ *Initialize*: Set $mu$ to the arithmetic mean of shapes (starting approximation)
+ *Repeat until convergence*:
  + Compute log maps: $bold(p)_i = "Log"_mu (S_i)$ for all shapes via LDDMM registration
  + Average in tangent space: $macron(bold(p)) = 1/K sum_i bold(p)_i$
  + Update atlas: $mu <- "Exp"_mu (alpha dot macron(bold(p)))$ via geodesic shooting
+ *Output*: The converged $mu$ is the Fréchet mean

The step size $alpha in (0, 1]$ controls the update magnitude per iteration.

== Tangent Space PCA

=== The Idea

Since the tangent space at the atlas *is* a vector space, we can perform standard PCA there @miller2015hamiltonian:

+ Compute all initial momenta: $bold(p)_i = "Log"_"atlas" (S_i)$ via LDDMM registration
+ Flatten to vectors: $bold(p)_i in RR^(N times 3) -> RR^(3N)$
+ Subtract mean: $tilde(bold(p))_i = bold(p)_i - macron(bold(p))$
+ SVD: $tilde(P) = U Sigma V^T$
+ Principal components: columns of $V$ (reshaped to $N times 3$)

=== Shape Synthesis

To generate a new shape from PCA coefficients $c = (c_1, ..., c_k)$:

$ bold(p) = macron(bold(p)) + sum_(j=1)^k c_j dot sqrt(lambda_j) dot bold(v)_j $

$ S = "Exp"_"atlas" (bold(p)) $

where $bold(v)_j$ are principal components, $lambda_j$ are eigenvalues, and $"Exp"$ is geodesic shooting.

=== Shape Projection

To project a new shape $S$ onto the PCA basis:

+ Compute log map: $bold(p) = "Log"_"atlas" (S)$ via LDDMM registration
+ Center: $tilde(bold(p)) = bold(p) - macron(bold(p))$
+ Project: $c = tilde(bold(p)) dot V$ (inner product with principal components)

=== Interpretation

- *Component 1*: Direction of maximum variance (e.g., femur length)
- *Component 2*: Second-most variance (e.g., head angle)
- *Coefficients*: "Coordinates" in the shape space

== LDDMM vs Linear PCA: Summary

=== Key Differences

#figure(
  table(
    columns: (auto, auto, auto),
    inset: 8pt,
    align: left,
    table.header([*Aspect*], [*Linear PCA*], [*LDDMM*]),
    [Shape representation], [Vector in $RR^(3N)$], [Point on $"Diff"(Omega)$ manifold],
    [Distance], [$norm(S_1 - S_2)_F$], [Geodesic (regularized path length)],
    [Mean], [$(1/K) sum_i S_i$], [Fréchet mean (iterative)],
    ["Momentum"], [Displacement: $S - mu$], [Log map via registration],
    [Interpolation], [$(1-t)S_1 + t S_2$], [Geodesic shooting],
    [Extrapolation], [May self-intersect], [Diffeomorphic (valid shapes)],
  ),
  caption: [Summary of differences between Linear PCA and LDDMM approaches.]
) <tab:pca-vs-lddmm>

=== Why Momenta $eq.not$ Displacements

Linear PCA uses displacements: $"momentum" = "target" - "source"$

This ignores the geometry of deformations. True LDDMM momenta are obtained by solving a registration problem that finds the smoothest diffeomorphism.

The momentum $bold(p)_0$ satisfies: $v_0 = K * bold(p)_0$ (velocity = kernel applied to momentum).

The kernel $K$ enforces smoothness---nearby points must move coherently.

=== Advantages of True LDDMM

+ *Topology preservation*: All interpolated and extrapolated shapes remain valid (no self-intersections)
+ *Physically plausible deformations*: Smooth, coherent transformations respecting anatomical constraints
+ *Proper statistics*: Analysis respects the curved geometry of shape space
+ *Large deformation handling*: No artifacts from linear approximation
+ *Consistent distance metric*: Geodesic distance is geometrically meaningful




