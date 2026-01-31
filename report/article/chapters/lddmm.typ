// LDDMM Theory and Implementation
// This chapter explains the mathematical foundations of our LDDMM implementation

#import "@preview/fletcher:0.5.8" as fletcher: diagram, node, edge

#figure(
  caption: []
)[
  #image("/resources/img/fishdarcy.png", width: 80%) 
]
In an altogether remarkable work, the scholar, biologist, and mathematician D’Arcy Wentworth Thompson (1860-1948) underscored the importance of environmental and physical factors (as opposed to heredity alone) in the morphogenesis of living beings. Since the shape of fish is more or less optimal, there is not an infinite number of distinct "plans," but rather a mere handful of original patterns which allow, through (non-trivial) deformations, the generation of all forms favored by evolution.
    
    To describe the anatomical variability of an observed family or population, it is therefore sufficient to provide a _reference template_ (arbitrarily complex, yet common to all observations) and the deformations required to map said template to the individuals. Complexity is thus decoupled into two intelligible components: a _reference image_, complex but fixed; and _deformations_ specific to the observed subjects, often simple enough to be described with few parameters.
    
    Remarkable fact: the diagrams above present the variability of fish shapes not as arbitrary displacements of skeletons, but as *coordinate changes, deformations of the ambient space*. It is upon this mathematical concept of _extrinsic_ deformation of space (as opposed to _intrinsic_ movements of fish particles) that Procrustean analysis and the “LDDMM” theory presented in this chapter rely.

    We will attempt to present the big ideas and algorithms of LDDMM for discrete shape spaces with point correspondance (commonly known as landmarks spaces) in an intuitive manner, preserving the most mathematical content as possible but surely making compromises. For full technical details, proofs and rigorous presentation, see @younes2010shapes, @beg2005computing , @miller2015hamiltonian, @joshi2000landmark . 
    This study will be centered on the application of LDDMM to human femurs, but the global theory is a general computational anatomy framework.

== Overview
*Large Deformation Diffeomorphic Metric Mapping* (LDDMM) is a mathematical framework for computing smooth, invertible transformations between shapes @beg2005computing. Unlike linear methods that treat shapes as vectors in Euclidean space, LDDMM treats shapes as points on a *Riemannian manifold*, i.e a curved locally Euclidean space, respecting the intrinsic geometry of the shape space.
Consider the space of all possible femur shapes $cal(S)$. There is no biological nor anatomical reason suggesting this should be a vector space---you cannot simply add two femurs and get a valid femur [SOURCE]. Studying large deformations with a Riemannian approach has been an efficient point of view to generate metrics between deformable objects, and to provide accurate, non ambiguous and smooth matchings between shapes. 

#figure(
  caption: [Curved space is not stable by linear operations, and requires a  metric aware of its geometry]
)[
  #image("/resources/img/sphere_geodesic.png", width: 50%) 
]
In addition to anatomical implausibility of linear transformations, the Euclidean distance poses other challenges to anatomical relevancy, as it treats all point movements equally : this is not the general case in nature, as some deformations physically "cost" more than others, rendering them less probable. The straight line $S_1 -> S_2$ is not the optimal deformation path on the shape manifold.

For statistical shape analysis, we need two fundamental objects:
- A *distance* between shapes (how different are two femurs?)
- A *linear space* for statistics (PCA requires vector operations)


LDDMM provides both through its Riemannian geometry. Crucially, the resulting mean and distances are geometrically meaningful and respect the physical constraints of anatomical deformations @younes2010shapes. We first present the broad theoretical plan, before diving into technical aspects of the construction aimed at making the problem both interpretable and computationally tractable.

Our goal is to build/learn a relevant metric on $op("Diff")(cal(S))$, the group of diffeomorphisms (smooth invertible transformations) of $cal(S)$. However, since $op("Diff")(cal(S))$ is curved and infinite dimensional, we cannot perform standard statistical analysis on it. Indeed, there exists a lot of different such ways to transform a shape into another. We thus need to restrict ourselves to some subspace of this group, building a diffeomorphism space that can reflect complex deformations while being computationally viable.

The general framework is as follows: we first construct a vector space (called "velocity field") whose norm defines the *cost* of an infinitesimal deformation; the diffeomorphisms enabling the deformation of our shapes are then obtained by integrating a flow equation.

This approach is powerful but still yields a bunch of different "velocity fields" leading to the same diffeomorphism : we thus focus on the *cheapest* ones w.r.t a general *energy* of the deformation derived from the cost of the velocity field.
 
One reason for this is it provides a *geodesic distance* on $cal(S)$, computed as the energy of the *cheapest* transformation from $S_1$ to $S_2$

The notion of geodesic is a generalization of the notion of a "straight line" where every step of the movement must lie on the manifold. Here, the "movement" is the deformation, and it belonging to $op("Diff")(cal(S))$ ensures this property. Geodesics are the "shortest" diffeormorphic paths (w.r.t to the energy) between two shapes. [SOURCE]

Using this distance, we can define a mean shape of our population of $K$ shapes, commonly referred to as *atlas* in Riemannian geometry.

$ macron(S) = arg min_S sum_(i=1)^K d^2 (S, S_i) $

One important result of LDDMM theory is that, given the right diffeomorphism space, focusing on geodesic deformations is not only good for defining a distance :

- It is plausible to believe that nature favors "cheap" deformations from a biological standpoint, analogous to the principle of least action. [SOURCE]

- Since our space of deformations is a manifold, it is locally Euclidean around any shape $S$, i.e. is a flat vector space called the *tangent space* at $S$, denoted $T_S cal(S)$.

- A fundamental theorem on the nature of geodesics establishes a local bijection between geodesic deformations originating from the atlas and their *initial momenta*—vectors with one component attached to every vertex of the shape, which thus live in the $3N$-dimensional subspace of the tangent space at the atlas $T_(macron(S)) cal(S)$.

The *initial momentum* $bold(p)_0$ at the atlas encodes "which direction and how far" to travel along a geodesic to reach a target shape. This bijection is realized through two fundamental maps:
- The *exponential map* $"Exp"_(macron(S)) : T_(macron(S)) cal(S) -> cal(S)$ sends an initial momentum to the shape reached by following the corresponding geodesic for unit time.
- The *logarithm map* $"Log"_(macron(S)) : cal(S) -> T_(macron(S)) cal(S)$ is its inverse, returning the initial momentum that generates a geodesic to a given shape.
Said in a more compact way, the following diagram commutes :

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 2em,
    align: horizon,

    // Left Column: Your Fletcher Diagram
    fletcher.diagram(
      spacing: (25mm, 20mm),
      node-stroke: 0.5pt,
      node-inset: 8pt,
      node((0, 0), $T_(macron(S)) cal(S)$, name: <tangent>),
      node((1, 0), $cal(S)$, name: <shape>),
      edge(<tangent>, <shape>, $"Exp"_(macron(S))$, "->", bend: 25deg),
      edge(<shape>, <tangent>, $"Log"_(macron(S))$, "->", bend: 25deg),
      node((0.5, 0.7), $bold(p)_0 |-> S = "Exp"_(macron(S))(bold(p)_0)$, stroke: none),
    ),

    // Right Column: The Image
    image("/resources/img/exp_and_log.png", width: 100%)
  ),
  caption: [The exponential and logarithm maps establish a local diffeomorphism between the tangent space at the atlas and the shape manifold.],
) <fig:exp-log-diagram>Beware this is only a *local* result, thus only valid for "small" deformations of the atlas, a limitation we precise and adress later.

This geodesic point of view thus yields us the two fundamental objects needed for statistical analysis. We can then perform standard PCA (linear or kernel) on $T_(macron(S)) cal(S)$ in order to understand the most probable deformations of the mean femur $macron(S)$. We call this *Tangent PCA* or *Principal Geodesic Analysis* (PGA).

Computationally, the most frequent operations to perform are $op("Exp")$ (called *geodesic shooting* ) and $op("Log")$ (called *registration*). Rendering these computations tractable is thus one important goal that influences the construction of the diffeomorphism space. 

== Diffeomorphism spaces in LDDM : General Principle 
In this section, we describe the general principle of the building of diffeomorphism spaces in the LDDMM framework.
This presentation is adapted from [SOURCE], which the interested reader should refer to for a more detailed presentation.
// Definitions for formatting theorems, etc.
#let definition(title, body) = block(width: 100%, inset: 8pt, fill: luma(245), stroke: (left: 2pt + black))[
  *#title* #body
]
#let theorem(title, body) = block(width: 100%, inset: 8pt, fill: luma(245), stroke: (left: 2pt + black))[
  *#title* #body
]
#let proposition(title, body) = block(width: 100%, inset: 8pt, fill: luma(245), stroke: (left: 2pt + black))[
  *#title* #body
]
#let remark(title, body) = block(width: 100%, inset: 8pt)[
  #title #body
]
#let example(title, body) = block(width: 100%, inset: 8pt)[
  #title #body
]

=== General framework : integrating the flow of vector fields
#definition("Definition 4.1.")[
  A vector space $V$ of vector fields on $RR^d$ is said to be _admissible_ if it satisfies the following conditions:
  
  1. $V$ is a Hilbert space. We denote its norm by $||dot||_V$ and its inner product by $angle.l dot, dot angle.r_V$.
  2. $(V, ||dot||_V)$ is continuously embedded in $(C_0^1(RR^d, RR^d), ||dot||_{1, infinity})$, the space of $C^1$ fields on $RR^d$ vanishing at infinity, along with their partial derivatives. There exists, therefore, a constant $c_V > 0$ such that:
  $ forall v in V, quad ||v||_(1, infinity) <= c_V ||v||_V $ <eq:13.1>
]

In fact, the diffeomorphisms we are about to define are viewed as solutions to a flow equation. We observe a vector field $v_t$ (modeling infinitesimal deformations) deforming our space over time. Upon reaching time $t$, the deformation defines a diffeomorphism. A theorem guarantees the existence and uniqueness of such solutions. However, we require additional hypotheses on our vector fields, specifically an $L^2$ control. #footnote[We actually only need a $L^1$ control for the theorem, but a lemma allows us to restrict our following study to vector fields with an $L^2$ control whilst not losing any generality.] 
#definition("Definition 4.2.")[
  We define the following space and norm:
  $ L_V^2 = L^2 ([0, 1], V) "endowed with" ||v||_(L_V^2) = sqrt(integral_0^1 ||v_t||_V^2 dif t) $ <eq:13.3>
]

#theorem("Theorem 4.1.")[
  Let $v in L_V^2$. For all $x in RR^d$, there exists a unique continuous mapping $t mapsto Phi_t^v(x)$ from $[0, 1]$ to $RR^d$ satisfying the flow equation:
  $ Phi_t^v (x) = x + integral_0^t v_s compose Phi_s^v (x) dif s $ <eq:13.6>
  Alternatively,
  $ Phi_0^v (x) = x quad "and" (diff Phi_t^v) / (diff t) (x) = v_t (Phi_t^v (x)) $ <eq:13.7>

]

The diffeomorphisms that will serve our purpose are thus the elements of the set:
$ cal(D)_V = { Phi_1^v | v in L_V^2 } $ <eq:13.8>

Indeed, it suffices to consider flows at time 1 via time renormalization. This definition of diffeomorphisms on our space allows us to establish a metric, which in turn will allow us to evaluate the cost associated with using a diffeomorphism to deform our space.

#proposition("Proposition 4.1.")[
  $cal(D)_V$ is a group and a complete space for the metric:
  $ d_V (id, Phi) = inf { ||v||_(L_V^2) | v in L_V^2, Phi_1^v = Phi } $ <eq:13.9>
  which can be extended by right-invariance:
  $ d_V (Phi, Psi) = d_V (id, Psi compose Phi^(-1)) $ <eq:13.10>
]

This metric is simply defined as the infimum of the norm of vector fields that deform the identity to $Phi(id)$.

If we take two diffeomorphisms in $cal(D)_V$, there exists a geodesic between them for the metric we have defined, i.e the infimum is actually a minimum.

#proposition("Proposition 4.2")[
  Let $Phi, Psi in cal(A)_V$.
  There exists $v in L_V^2$ such that $Phi_1^v = Phi compose Psi^(-1)$ and $d(Phi, Psi) = ||v||_(L_V^2) $. 
]

We say that $cal(D)_V$ is the group of diffeomorphisms modeled on $V$ starting from the identity. \
In the "limit" case where $V$ is the space of constant fields, identified with $RR^m$ equipped with the Euclidean metric, $cal(A)_V$ is simply the space of translations equipped with the natural Euclidean distance.

By considering a more flexible space $V$, we cause the number of degrees of freedom—and thus the complexity of the space of diffeomorphisms $cal(A)_V$—to explode. The entire objective of this exposition will be to see how to choose $V$ to keep the problem _reasonable_ and numerically solvable.

== The Matching problem 
We have defined a metric on a general diffeomorphism space that deforms our shape space $cal(S)$.
A linear algebra theorem guarantees the existence of a diffeomorphism $Phi in cal(D)_V$ such that $Phi(S_1) = S_2$ for arbitrary shapes $S_1, S_2$ To derive a distance on $cal(S)$, we could thus theoretically just define $ d_V (S_1,S_2) = d_V (id, Phi) = min_(v in L^2_V) sqrt(integral_0^1 norm(v_t)_V^2 dif t). $

In practice, finding the $Phi$ that exactly maps $S_1$ to $S_2$ is too hard, we thus relax the problem by defining a naive *matching* distance, which we can do because our shape space is a landmark space. #footnote[This becomes way harder in non-landmark spaces, and is done via Varifolds. See [SOURCE] for more details. ] \
#definition("Matching distance.")[
Let $Phi in cal(D)_V$, $S_1 = {x_i, i in bracket.stroked 1, n bracket.stroked.r} in cal(S)$ and $S_2 in cal(S)$. The matching distance between $S_2$ and $Phi(S_1) = {y_i, i in bracket.stroked 1, n bracket.stroked.r} $ is defined by:
  $ d_cal(S)(Phi) = norm(phi(S_1) - S_2)_2 ^2 = sum_(i=1)^(n) abs(y_i - x_i)^2  $
]
This matching (squared euclidean) distance is not a relevant quantity in general as already discussed, but can be used to assess if two shapes are "very" close or not.

We are now ready to properly define a relevant *energy* (cost) of the transformation from $S_1$ to $S_2$, by minimizing the cost of $Phi$ and relaxing the matching :

#definition("Matching problem")[Let $lambda > 0$. The matching problem is defined by :

$ op("Minimize") quad quad  E(Phi) =  lambda d_V ( id , Phi) + d_cal(S) (Phi) "over" cal(D)_V $

Which is equivalent to minimizing the following functional over $L^2_V$ :
$ E(v) = lambda integral_0^1 norm(v_t)_V^2 dif t + d_cal(S) (Phi^v_1). $] <matching-1>

To solve this matching problem, we have an existence theorem at our disposal, which assures us that this endeavor is not futile and that there effectively exist diffeomorphisms allowing us, in practice, to map one shape to another in an optimal manner.

In practice, the task consists of finding a diffeomorphism $Phi$ that deforms a shape $S_1$ into a shape $Phi(S_1)$ close to $S_2$. The optimal shape $Phi(S_1)$ can then be viewed as a _barycenter_ of the points $S_1$ — with weight $lambda$ — and $S_2$ — with weight $1$.

The real number $lambda$ is a _regularization_ weight, which compels $Phi(S_1)$ to remain within a neighborhood of $S_1$: one often selects $0 < lambda << 1$, so as to obtain a model $Phi(S_1)$ close to the observation $S_2$, yet remaining at a "finite" deformation distance from the shape $S_1$.
== Building the right vector field space

The matching problem as expressed in [REF] is over $L^2_V$, which can _a priori_ be infinite dimensional. Our goal is thus to build $V$ such that we can _reduce_ #footnote[_Reduction_ is the process of proving that the optimization of a functional over a larger space can be done over a smaller space] the problem to a finite dimensional setting, making it numerically solvable. In this section, we define $V$ as a *Reproducing Kernel Hilbert Space* (RHKS) induced by a positive definite kernel $k$. In this context, we perform a first reduction step to the space of vector fields which are null outside of the shape's region of $RR^3$. We then reinterpret our matching problem in terms of *Riemannian geometry*, ultimately yielding a _stunning_ reduction of the problem to *initial momenta* in $RR^(3N)$.
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




