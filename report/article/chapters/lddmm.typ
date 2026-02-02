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

    We will attempt to present the big ideas and algorithms of LDDMM for discrete shape spaces with point correspondence (commonly known as landmark spaces) in an intuitive manner, preserving the most mathematical content as possible but surely making compromises. For full technical details, proofs and rigorous presentation, see @younes2010shapes, @beg2005computing , @miller2015hamiltonian, @joshi2000landmark . 
    This study will be centered on the application of LDDMM to human femurs, but the global theory is a general computational anatomy framework.

== Overview
*Large Deformation Diffeomorphic Metric Mapping* (LDDMM) is a mathematical framework for computing smooth, invertible transformations between shapes @beg2005computing. Unlike linear methods that treat shapes as vectors in Euclidean space, LDDMM treats shapes as points on a *Riemannian manifold*, i.e., a curved, locally Euclidean space, respecting the intrinsic geometry of the shape space.
Consider the space of all possible femur shapes $cal(S)$. There is no biological nor anatomical reason suggesting this should be a vector space—you cannot simply add two femurs and get a valid femur @younes2010shapes. In the landmark setting, we model shapes as elements of $(RR^3)^n$, but the admissible shapes form a nonlinear embedded submanifold $cal(S) subset (RR^3)^n$. Let $A = RR^3$ denote the ambient space. Studying large deformations with a Riemannian approach has been an efficient point of view to generate metrics between deformable objects, and to provide accurate, unambiguous and smooth matchings between shapes. 

#figure(
  caption: [Curved space is not stable by linear operations, and requires a  metric aware of its geometry]
)[
  #image("/resources/img/sphere_geodesic.png", width: 50%) 
]
In addition to anatomical implausibility of linear transformations, the Euclidean distance poses other challenges to anatomical relevancy, as it treats all point movements equally: this is not the general case in nature, as some deformations physically "cost" more than others, rendering them less probable. The straight line $S_0 -> S_1$ is not the optimal deformation path on the shape manifold.

For statistical shape analysis, we need two fundamental objects:
- A *distance* between shapes (how different are two femurs?)
- A *linear space* for statistics (PCA requires vector operations)


LDDMM provides both through its Riemannian geometry. Crucially, the resulting mean and distances are geometrically meaningful and respect the physical constraints of anatomical deformations @younes2010shapes. We first present the broad theoretical plan, before diving into technical aspects of the construction aimed at making the problem both interpretable and computationally tractable.

Our goal is to build/learn a relevant metric on $op("Diff")(A)$, the group of diffeomorphisms (smooth invertible transformations) of the ambient space $A$. However, since $op("Diff")(A)$ is curved and infinite dimensional, we cannot perform standard statistical analysis on it. Indeed, there exists a lot of different such ways to transform a shape into another. We thus need to restrict ourselves to some subspace of this group, building a diffeomorphism space that can reflect complex deformations while being computationally viable.

The general framework is as follows: we first construct a vector space (called "velocity field") whose norm defines the *cost* of an infinitesimal deformation; the diffeomorphisms enabling the deformation of our shapes are then obtained by integrating a flow equation.

This approach is powerful but still yields a bunch of different time-dependent "velocity fields" leading to the same diffeomorphism: we thus focus on the *cheapest* ones with respect to a general *energy* of the deformation derived from the cost of the velocity field.
 
One reason for this is it provides a *geodesic distance* on $cal(S)$, computed as the energy of the *cheapest* transformation from $S_0$ to $S_1$.

The notion of geodesic is a generalization of the notion of a "straight line" where every step of the movement must lie on the manifold. Here, the "movement" is the deformation, and its belonging to $op("Diff")(A)$ ensures this property. Geodesics are locally length-minimizing diffeomorphic paths (with respect to the energy) between two shapes; we return to the global subtleties later, see @younes2010shapes.

Using this distance, we can define a mean shape of our population of $K$ shapes, commonly referred to as *atlas* in Riemannian geometry.

$ macron(S) = arg min_S sum_(i=1)^K d^2 (S, S_i) . $

One important result of LDDMM theory is that, given the right diffeomorphism space, focusing on geodesic deformations is not only good for defining a distance:

- It is plausible to believe that nature favors "cheap" deformations from a biological standpoint, analogous to the principle of least action, see @miller2015hamiltonian.

- Since our space of deformations is a manifold, it is locally Euclidean around any shape $S$, i.e., it is a flat vector space called the *tangent space* at $S$, denoted $T_S cal(S)$.

- A fundamental theorem on the nature of geodesics establishes a local bijection between geodesic deformations originating from the atlas and their *initial momenta*—vectors with one component attached to every vertex of the shape, which thus live in the $3n$-dimensional cotangent space at the atlas $T_macron(S)^* cal(S)$. Through the cometric, these momenta correspond to initial velocities in the tangent space.

The *initial momentum* $P_0$ at the atlas encodes "which direction and how far" to travel along a geodesic to reach a target shape. Via the cometric, it corresponds to an initial velocity in the tangent space. This bijection is realized through two fundamental maps:
- The *exponential map* $"Exp"_(macron(S)) : T_macron(S) cal(S) -> cal(S)$ sends an initial velocity to the shape reached by following the corresponding geodesic for unit time.
- The *logarithm map* $"Log"_(macron(S)) : cal(S) -> T_macron(S) cal(S)$ is its inverse, returning the initial velocity that generates a geodesic to a given shape.
Said in a more compact way, the following diagram commutes:

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
      node((0, 0), $T_macron(S) cal(S)$, name: <tangent>),
      node((1, 0), $cal(S)$, name: <shape>),
      edge(<tangent>, <shape>, $"Exp"_(macron(S))$, "->", bend: 25deg),
      edge(<shape>, <tangent>, $"Log"_(macron(S))$, "->", bend: 25deg),
      node((0.5, 0.7), $v_0 |-> S = "Exp"_(macron(S))(v_0)$, stroke: none),
    ),

    // Right Column: The Image
    image("/resources/img/exp_and_log.png", width: 100%)
  ),
  caption: [The exponential and logarithm maps establish a local diffeomorphism between the tangent space at the atlas and the shape manifold.],
  ) <fig:exp-log-diagram>



This geodesic point of view thus yields us the two fundamental objects needed for statistical analysis. We can then perform standard PCA (linear or kernel) on $T_macron(S) cal(S)$ in order to understand the most probable deformations of the mean femur $macron(S)$. We call this *Tangent PCA* or *Principal Geodesic Analysis* (PGA).

Computationally, the most frequent operations to perform are $op("Exp")$ (called *geodesic shooting* ) and $op("Log")$ (called *registration*). Rendering these computations tractable is thus one important goal that influences the construction of the diffeomorphism space. 

== Diffeomorphism spaces in LDDMM: General Principle 
In this section, we describe the general principle of the building of diffeomorphism spaces in the LDDMM framework.
This presentation is adapted from @younes2010shapes, which the interested reader should refer to for a more detailed presentation.
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

=== General framework: integrating the flow of vector fields
#definition("Definition.")[
  A vector space $V$ of vector fields on $RR^d$ is said to be _admissible_ if it satisfies the following conditions.
  
  1. $V$ is a Hilbert space. We denote its norm by $norm(dot)_V$ and its inner product by $angle.l dot, dot angle.r_V$.
  2. $(V, norm(dot)_V)$ is continuously embedded in $(C_0^1(RR^d, RR^d), norm(dot)_(1, infinity))$, the space of $C^1$ fields on $RR^d$ vanishing at infinity, along with their partial derivatives. There exists, therefore, a constant $c_V > 0$ such that:
  $ forall v in V, quad norm(v)_(1, infinity) <= c_V norm(v)_V . $
]

The diffeomorphisms we are about to define are viewed as solutions to a flow equation. We observe a vector field $v_t$ (modeling infinitesimal deformations) deforming our space over time. Upon reaching time $t$, the deformation defines a diffeomorphism. A theorem guarantees the existence and uniqueness of such solutions. However, we require additional hypotheses on our vector fields, specifically an $L^2$ control. #footnote[We actually only need a $L^1$ control for the theorem, but a lemma allows us to restrict our following study to vector fields with an $L^2$ control whilst not losing any generality.] 
#definition("Definition.")[
  We define the following space and norm.
  $ L_V^2 = L^2 ([0, 1], V) "endowed with" norm(v)_(L_V^2) = sqrt(integral_0^1 norm(v_t)_V^2 dif t) . $
]
The elements of $L_V^2$ are time-dependent vector fields $v : t mapsto v(t)$ where we denote $v(t) = v_t$, for $t in [0,1]$ and $v_t in V$.
#theorem("Theorem.")[
  Let $v in L_V^2$. For all $x in RR^d$, there exists a unique continuous mapping $t mapsto Phi_t^v(x)$ from $[0, 1]$ to $RR^d$ satisfying the flow equation.
  $ Phi_t^v (x) = x + integral_0^t v_s compose Phi_s^v (x) dif s . $
  Alternatively,
  $ Phi_0^v (x) = x quad "and" (diff Phi_t^v) / (diff t) (x) = v_t (Phi_t^v (x)) . $
]

The diffeomorphisms that will serve our purpose are thus the elements of the set:
$ cal(D)_V = { Phi_1^v | v in L_V^2 } . $

Indeed, it suffices to consider flows at time 1 via time renormalization. This definition of diffeomorphisms on our space allows us to establish a metric, which in turn will allow us to evaluate the cost associated with using a diffeomorphism to deform our space.

#proposition("Proposition.")[
  $cal(D)_V$ is a group and a complete space for the metric.
  $ d_V (id, Phi) = inf { norm(v)_(L_V^2) | v in L_V^2, Phi_1^v = Phi } . $
  Which can be extended by right-invariance:
  $ d_V (Phi, Psi) = d_V (id, Psi compose Phi^(-1)) . $
]

This metric is simply defined as the infimum of the norm of vector fields that deform the identity to $Phi(id)$.

If we take two diffeomorphisms in $cal(D)_V$, there exists a geodesic between them for the metric we have defined, i.e., the infimum is actually a minimum.

#proposition("Proposition.")[
  Let $Phi, Psi in cal(D)_V$.
  There exists $v in L_V^2$ such that $Phi_1^v = Phi compose Psi^(-1)$ and $d_V (Phi, Psi) = norm(v)_(L_V^2) . $
]

We say that $cal(D)_V$ is the group of diffeomorphisms modeled on $V$ starting from the identity. \
In the "limit" case where $V$ is the space of constant fields, identified with $RR^m$ equipped with the Euclidean metric, $cal(D)_V$ is simply the space of translations equipped with the natural Euclidean distance.

By considering a more flexible space $V$, we cause the number of degrees of freedom—and thus the complexity of the space of diffeomorphisms $cal(D)_V$—to explode. The entire objective of this exposition will be to see how to choose $V$ to keep the problem _reasonable_ and numerically solvable.

#figure(
  image("/resources/img/extrinsic_def_ex.png", width: auto, height: auto),
  caption: [Deformation of a surface in another under the action of a diffeomorphism. Step by step, we see the ambient grid deform under the action of a vector field $v_t$.],
) <extrinsic>

== The Matching problem 
We have defined a metric on a general diffeomorphism space that deforms our shape space $cal(S)$. Considering arbitrary shapes $S_0, S_1 in cal(S)$, a theorem guarantees the existence of a diffeomorphism $Phi in cal(D)_V$ such that $Phi(S_0) = S_1$. We could thus theoretically define a distance on $cal(S)$ in the following way: $ d_V (S_0,S_1) = d_V (id, Phi) = min_(v in L^2_V) sqrt(integral_0^1 norm(v_t)_V^2 dif t) . $

In practice, finding the $Phi$ that exactly maps $S_0$ to $S_1$ is too hard, we thus relax the problem by defining a naive *matching* distance, which we can do because our shape space is a landmark space. #footnote[This becomes way harder in non-landmark spaces, and is done via Varifolds. See @younes2010shapes for more details. ] \
#definition("Matching distance.")[
Let $Phi in cal(D)_V$, $S_0 = {x_i, i in bracket.stroked 1, n bracket.stroked.r} in cal(S)$ and $S_1 = {z_i, i in bracket.stroked 1, n bracket.stroked.r} in cal(S)$. The matching distance between $S_1$ and $Phi(S_0) = {y_i, i in bracket.stroked 1, n bracket.stroked.r} $ is defined by:
  $ d_cal(S)(Phi) = norm(Phi(S_0) - S_1)_2^2 = sum_(i=1)^n abs(y_i - z_i)^2 . $
]
This matching ($L^2$, squared Euclidean) distance is not a relevant quantity in general as already discussed, but can be used to assess if two shapes are "very" close or not.

We are now ready to properly define a relevant *energy* (cost) of the transformation from $S_0$ to $S_1$, by minimizing the cost of $Phi$ and relaxing the matching:

#definition("Matching problem.")[Let $lambda > 0$. The matching problem is defined by.

$ op("Minimize") quad quad  E(Phi) =  lambda d_V ( id , Phi) + d_cal(S) (Phi) "over" cal(D)_V . $

Which is equivalent to minimizing the following functional over $L^2_V$:
$ E(v) = lambda integral_0^1 norm(v_t)_V^2 dif t + d_cal(S) (Phi^v_1) . $] <def:matching-problem>

To solve this matching problem, we have an existence theorem at our disposal, which assures us that this endeavor is not futile and that there effectively exist diffeomorphisms allowing us, in practice, to map one shape to another in an optimal manner.

In practice, the task consists of finding a diffeomorphism $Phi$ that deforms a shape $S_0$ into a shape $Phi(S_0)$ close to $S_1$. The optimal shape $Phi(S_0)$ can then be viewed as a _barycenter_ of the points $S_0$ — with weight $lambda$ — and $S_1$ — with weight $1$.

The real number $lambda$ is a _regularization_ weight, which compels $Phi(S_0)$ to remain within a neighborhood of $S_0$: one often selects $0 < lambda << 1$, so as to obtain a model $Phi(S_0)$ close to the observation $S_1$, yet remaining at a "finite" deformation distance from the shape $S_0$.
== Kernel Methods : Efficient and Interpretable Vector Fields

The matching problem as expressed previously /* @def:matching-problem */ is over $L^2_V$, which can _a priori_ be infinite dimensional. Our goal is thus to build $V$ such that we can _reduce_ #footnote[_Reduction_ is the process of proving that the optimization of a functional over a larger space can be done over a smaller space] the problem to a finite dimensional setting, making it numerically solvable. In this section, we define $V$ as the *Reproducing Kernel Hilbert Space* (RKHS) induced by a positive definite kernel $k$. In this context, we perform a first reduction step allowing us to only focus on defining the deformation at the landmarks. We then introduce the *Gaussian kernel*, used in practice. Finally, we reinterpret our matching problem in terms of *Riemannian geometry*, ultimately yielding a _stunning_ reduction of the problem to *initial momenta* in $RR^(3n)$.


=== Reproducing space and kernels

In the following section, $A = RR^3$ is our ambient space, thus vectors of dimension $n$ of $A$ represent shapes with $n$ landmarks. We adopt the notation $q$ for positions (landmarks) in $A$ and $p$ for momenta or vectors, in order to anticipate the Hamiltonian formalism introduced later.

#definition("RKHS.")[
  Let $H$ a Hilbert space of vector fields $A -> RR^3$.
  $H$ is called a _Reproducing Kernel Hilbert Space_ (or RKHS) when the linear functionals \ $delta_q^p : f in H mapsto f(q) dot p$, where $q in A$ and $p in RR^3$, are continuous.
]

We can then apply the Riesz theorem to $delta_q^p$ and define the reproducing kernel of an RKHS:

#definition("Reproducing kernel.")[
  Let H be an RKHS. The _reproducing kernel_ of $H$ is the mapping \ $k_H : A^2 -> cal(L)(RR^3)$ such that:
  $ forall q in A, k_H (q, dot) p = K_H delta_q^p $  meaning that $k_H (q, dot) p$ is the unique element of $H$ such that:
  $ forall h in H, delta_q^p (h) = angle.l k_H (q, dot) p, h angle.r_H = h(q) dot p $ ]


#definition("Positive kernel.")[
Let $k : A^2 -> cal(L)(RR^3)$. For any shape $S = (q_1, dots, q_n) in A^n$, we define the associated kernel matrix $K_S$ as the following square block matrix of size $3n$:

  $ K_S = mat(delim: "(",
    k(q_1, q_1), k(q_1, q_2), dots, k(q_1, q_n);
    k(q_2, q_1), k(q_2, q_2), dots, k(q_2, q_n);
    dots.v, dots.v, dots.down, dots.v;
    k(q_n, q_1), k(q_n, q_2), dots, k(q_n, q_n)
  ) . $

  where each entry $k(q_i, q_j)$ is a $3 times 3$ matrix.
  The map $k$ is said to be a *positive kernel* if for every stacked vector of momenta $P = (p_1, dots, p_n)^T in (RR^3)^n$, the quadratic form defined by $K_S$ is non-negative:
  
  $ P^T K_S P >= 0 <=>  sum_(i, j) (p_i)^T k(q_i, q_j) p_j >= 0 . $
]

One can show the equivalence between the notions of reproducing kernel and positive kernel.

#theorem("Theorem.")[
  1. The reproducing kernel of an RKHS is a positive kernel.
  
  2. For any positive kernel $k$ on $A$ , there exists a unique RKHS included in $(RR^3)^A$ whose reproducing kernel is $k$.
]


The main appeal of reproducing kernels is they provide a simple method for defining admissible vector field spaces that are interpretable and algorithmically efficient. We first choose the positive kernel $k$ then define $V$ as the unique RKHS corresponding to $k$. For Riemannian interpretation purposes, we consider strictly positive kernels in practice, where $P^T K_S P > 0$, so that $K_S$ is invertible.

=== Gaussian Kernel

In practice, the choice of the kernel $k$ determines the nature of the deformations allowed by our model. The most common choice in computational anatomy is the **Isotropic Gaussian Kernel**.

It is defined by a scalar Gaussian curve acting diagonally on the ambient space vectors $x, y in A = RR^3$:

$ k_sigma (x, y) = exp( -norm(x - y)^2 / sigma^2 ) dot I_3 $

where $I_d$ is the identity matrix in $RR^d$ and $sigma$ is a scale parameter. The use of the identity matrix ensures *isotropy*: the cost of moving a point is independent of the direction of movement.

We later interpret this kernel as defining a Riemannian metric on $cal(S)$, and go over more details about the impact of a specific kernel.


== Spatial Reduction of the Matching problem
Consider a fixed time $t in [0, 1]$, and suppose the current positions of our $n$ landmarks forming the shape $S$ are $q_1, dots, q_n$.  \ Physically, the term $k(x, q_i) p_i$ represents the velocity field generated in the entire space by a single "push" (momentum $p_i$) applied at the point $q_i$. 
- If we push on a landmark $q_i$, the kernel acts as a transmission function, dragging the surrounding space along with it.
- The "shape" of this drag is determined by the kernel function.

Therefore, the subspace spanned by these kernels:<def:vs>
$ V_S = op("span")({k(dot, q_i) p_i | i=1..n, p_i in RR^3}) . $
represents the set of all infinitesimal deformations that can be constructed *solely* by pushing on the landmarks at time $t$.

We recall that at each time step, our goal is to find the vector field $v_t$ that moves our landmarks $q_i$ while minimizing the infinitesimal deformation energy $norm(v_t)_V^2$. #footnote[ Indeed, this implies a minimization of the total energy $integral_(0)^(1) norm(v_t)_V^2 dif t $.]

Consider any candidate vector field $v_t in V$. We can decompose it into two orthogonal components:
$ v_t = v_t^S + v_t^(perp) . $
where $v_t^S in V_S$ is built from the kernels applied at the landmarks, and $v_t^(perp)$ is orthogonal to them.

The crucial insight comes from the *Reproducing Property* of the RKHS. For any momentum $p$, the inner product with the kernel evaluates the field at the landmark:
$ angle.l v_t^(perp), k(dot, q_i) p angle.r_V = p dot v_t^(perp)(q_i) . $

Since $v_t^(perp)$ is orthogonal to the kernel, it is orthogonal to the landmark, i.e., $v_t^(perp)(q_i) = 0$.
 
The component $v_t^(perp)$ does not help move the landmarks, yet still adds to the total cost $norm(v_t)_V^2 = norm(v_t^S)_V^2 + norm(v_t^(perp))_V^2$. To minimize cost, we must therefore set $v_t^(perp) = 0$.

This reasoning tells us that the optimal deformation of the whole space is completely determined by the interaction of $n$ "bumps" centered at our landmarks, which yields a powerful reduction of our matching problem.

#theorem("Theorem.")[
  Let $S = {q_1, dots, q_n}$ denote the current landmarks. The vector field $v_t in V$ which interpolates specific velocities $v_t(q_i)$ with minimal norm is a linear combination of the kernel centered at the landmarks:
  
  $ forall x in A  , quad  v_t(x) = sum_(i=1)^n k(x, q_i) p_i . $
  
  where the vectors $p_i in RR^3$ act as *momenta* determining the strength and direction of the local deformation.
]
Crucially, the finite dimensional nature of $L^2_(V, S)$ makes for a simple computation of the norm:

$ norm(v_t)_V^2 = sum_(i  , j in bracket.stroked 1, n bracket.stroked.r ) p_j^T k(q_i,q_j)p_i  . $

#theorem("Corollary.")[
  The optimal solution of the problem
  $ op("Minimize") quad E(v) = lambda integral_0^1 norm(v_t)_V^2 dif t + d_cal(S) (Phi^v_1) . $
  can be searched over $L_(V,S)^2 = { v in L_V^2, forall t in [0, 1], v_t in V_(Phi_t^v (S)) }$, space of dimension $3n$.

]
At each time step, instead of solving a PDE for a complex function $v(x)$, we have reduced the problem to finding the optimal momenta $p_i$ for each landmark. The global flow is simply the superposition of these local kernel influences. More precisely, for any arbitrary point $x$ in our ambient space $A$ (e.g., a point of a femur mesh), the flow equation becomes:

$ (partial Phi_t^v) / (partial t) (x) = v_t (Phi_t^v (x)) = sum_(j=1)^n k(Phi_t^v (x), q_j) p_j . $

This shows that the deformation of the entire space is "interpolated" from the momenta $p_j$ of the driving landmarks.


If we track the landmarks themselves, denoting $q_i(t) = Phi_t^(v) (q_i (0))$ and $S(t) = (q_1(t), dots, q_n (t))$. Defining the stacked momentum $P(t) = (p_1(t), dots, p_n (t))^T$, the landmark dynamics can be written compactly as:<sec:landmark-flow>

$ (partial S) / (partial t) = K_(S(t)) P(t) . $

This important reduction step has made the problem somewhat #footnote[This is really theoretical, in the sense that you can write algorithms that perform this optimization. In practice, this problem is still way too complex to be solved in a reasonable amount of time.] computationally tractable since we are optimizing over a finite set of parameters, we can compute $ d_cal(S) (Phi^v_1)$ by integrating the flow equation, and easily compute $norm(v_t)_V^2$, hence $E(v)$ as a whole. \
However, a Riemannian geometric view of our dynamical system will shed light on the true nature of our problem, leading us to the most significant and _beautiful_ reduction step.


== Riemannian Geometry of Landmarks
=== Introduction

In this section, we intuitively introduce the fundamental concepts of Riemannian Geometry, the study of smooth curved spaces called *manifolds*. For a comprehensive introduction to this field of study, see @younes2010shapes. 

It is an abstract model that captures the essence of all possible spaces having the same intrinsic geometry. By the “essence” we mean
knowledge of the distance between any two points, for this and this alone determines the intrinsic
geometry. In fact, it is sufficient to have a rule for the inﬁnitesimal distance between neighbouring points. This rule is
called the *metric*. Given this metric, we may determine the length of any curve as an infinite sum
(i.e., integral) of the infinitesimal segments into which it may be divided.
We can then define a distance using the shortest paths between points on the manifold.
These shortest paths on a curved space are the equivalent of straight lines in the plane, they are called *geodesics*. Thus, to use this new word, we may say that geodesics in Euclidean space are straight lines, and geodesics on the sphere are
great circles.


#figure(
  caption: [Euclidean distance and distance on the sphere as a Riemannian manifold]
)[
  #image("/resources/img/sphere_geodesic.png", width: 50%) 
]

However, this _length-minimizing_ definition of geodesics is subtle, because even on the sphere, we see that nonantipodal points are connected by two arcs of the great circle passing through them:
the short one (which is the shortest route) and the long one. Yet the long arc is every bit as much _straight_, thus a
geodesic, as the short one. We thus provide a purely _local_ characterization of geodesics.

One key property of manifolds is they are locally Euclidean, i.e., look like flat space around any point when _zooming_ enough. This is formalized by the notion of *tangent space*, the flat space hugging our manifold exactly at a point.

#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    image("/resources/img/tangent_1.png", width: 100%),
    image("/resources/img/tangent_2.png", width: 100%),
  ),
  caption: [The tangent space at a point of the sphere is the plane touching the sphere at that single point, containing all possible directions in which one can instantaneously move starting from there.],
)

We experience this everyday walking on our beautiful planet Earth. To us, going "straight" is equivalent to the definition in flat space, i.e., walking without ever _turning_. As we live *on* the surface of Earth, thus do not see its curvature, pursuing this local straight behavior yields a global geodesic.
The only parameter that determines the particular geodesic we trace and the time it takes is our initial *velocity* vector. Indeed, once we are set on our path, flowing on the surface following the local rule of not turning is enough to characterize a unique geodesic. Conversely, provided we stay close enough to the source point, the unique geodesic connecting given source and target points is fully characterized by the initial velocity starting from the source.<sec:initial-velocity> 

This leads to another interpretation of the tangent space. Consider (conveniently) a Riemannian manifold $cal(S)$ and a point $macron(S)$. The tangent space at $macron(S)$, denoted $T_macron(S) cal(S)$ is the space of all possible velocity vectors a being moving *on* the surface can follow when it is in $macron(S)$. Armed with this insight, one can formalize the velocity $<->$ geodesic relationship.

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
      node((0, 0), $T_macron(S) cal(S)$, name: <tangent>),
      node((1, 0), $cal(S)$, name: <shape>),
      edge(<tangent>, <shape>, $"Exp"_(macron(S))$, "->", bend: 25deg),
      edge(<shape>, <tangent>, $"Log"_(macron(S))$, "->", bend: 25deg),
      node((0.5, 0.7), $v_0 |-> S = "Exp"_(macron(S))(v_0)$, stroke: none),
    ),

    // Right Column: The Image
    image("/resources/img/exp_and_log.png", width: 100%)
  ),
  caption: [The exponential and logarithm maps establish a local diffeomorphism between the tangent space at a point and the manifold.],
) <local-bijection>

- The *exponential map* $"Exp"_(macron(S)) : T_macron(S) cal(S) -> cal(S)$ sends an initial velocity vector to the point reached by following the corresponding geodesic for unit time.
- The *logarithm map* $"Log"_(macron(S)) : cal(S) -> T_macron(S) cal(S)$ is its inverse, returning the initial velocity vector that generates a geodesic to a given point.

As shown earlier by the example of antipodal points on the sphere, this result is only true *locally*: if the target and source points are too far apart, there can be multiple geodesics connecting them, thus multiple potential velocity vectors, yielding a multivalued $op("Log")$ map.

=== The Landmarks Manifold 

We interpret the set of all possible landmark configurations, i.e., our shape space $cal(S)$ as a smooth manifold embedded in $(RR^3)^n$.
To define the geometry, we must distinguish between the deformation itself and the force generating it. Let $S in cal(S)$ be a shape state (which we later interpret as the shape attained at time $t in [0,1]$ of a trajectory in $cal(S)$).

The tangent space $T_S cal(S)$ is the space of all possible instantaneous deformations of the shape $S$. An element $v_t in T_S cal(S)$ is a vector field restricted to the landmarks, i.e., a collection of velocity vectors attached to each point, which we can think of as a global velocity vector attached to the shape:
$ v_t = (v_(t,1), dots, v_(t,n)) in (RR^3)^n . $

This is exactly the definition on $V_S$ /* @def:vs */, which is the infinitesimal deformation space we are optimizing over thanks to our last reduction step.
The *Cotangent Space* $T_S^* cal(S)$ is the dual space of the tangent space. It contains the linear forms acting on velocities.
An element $P(t) in T_S^* cal(S)$ is called the *momentum* (or co-vector). In our context, it represents the "force" or "constraints" applied to the landmarks to drive the deformation. Like velocity, it is numerically represented as a stacked vector in $(RR^3)^n$, with one component attached to each landmark.


In standard Euclidean space, velocity and momentum are often identified ($v_t = P(t)$). In a curved space, they are distinct, and linked by the physics / metric / curvature of the medium.

At first, it seems unintuitive to introduce this space and not do everything with velocity in the classical tangent space. In practice, the kernel's action is best interpreted with regard to momentum. Please note that both notions are completely equivalent, i.e., $T_S^* cal(S) tilde.eq T_S cal(S)$.

The kernel matrix $K_S$, defined block-wise by $K_(i,j) = k(q_i, q_j)$, acts as the **Cometric** (inverse metric) on the manifold. It maps momentum to velocity:

$ v_t = K_S P(t) quad <==> quad v_(t,i) = sum_(j=1)^n k(q_i, q_j) p_j (t) . $


- The momentum $P$ is the control variable (the input "push").
- The kernel $K_S$ is the transmission mechanism (smoothing/correlation).
- The velocity $v_t$ is the resulting deformation (the output).

A Riemannian metric $g_S: (T_S cal(S))^2 mapsto RR$ is an inner product on the tangent space, which allows for norms, distances and angles in the tangent space, thus in the neighborhood of $cal(S)$. It defines the local geometry of the manifold and is expressed as:

$ g_S (u, v) = u^T M_S v $

with $M_S$ a certain matrix.

We want to define a Riemannian metric on $cal(S)$ such that the associated norm of vector fields (elements of $V_S tilde.eq T_S cal(S)$) is the norm on $V$, which measures the cost of the associated deformation:
$ g_S(v_t, v_t) = norm(v_t)_V^2 = P(t)^T K_S P(t) = v_t^T K_S^(-1) v_t $

Consequently, $M_S = K_S^(-1)$ and $K_S$ is called the inverse metric (or cometric).

=== Interpreting the Gaussian Kernel as a Riemannian metric

We recall the definition of the Gaussian kernel $k_sigma$, for $x, y in A = RR^3 "and" sigma > 0$:

$ k_sigma (x, y) = exp( - norm( x - y)^2 / sigma^2 ) dot I_d $


Consider a shape consisting of two nearby points $q_1, q_2$ and denote $k = k_sigma (q_1, q_2)$ their Gaussian correlation.

The inverse kernel matrix is $ K^(-1) = 1/(1-k^2) mat(1, -k; -k, 1) $

Let us compare the cost of moving the points together against the cost of breaking them apart.
We define a translation vector field $v_T = 1/sqrt(2) vec(1, 1)$ and a tearing vector field $v_D = 1/sqrt(2) vec(1, -1)$.
    $ norm(v_T)_V^2 =  1 / (1 + k)   quad   "and" quad norm(v_D)_V^2= 1 / (1 - k) $


When $sigma$ is large, the bell surface widens and the points get strongly correlated, $k approx 1$.
The cost of translation is bounded, but the cost of tearing explodes to infinity. The metric rewards nearby points moving together: the shape is locally solid.
  
Conversely, when $sigma$ is small, $k approx 0$ and the costs are identical. The metric allows points to move in opposite directions as easily as they move together: the shape locally acts like a fluid.
#figure(
  grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #image("/resources/img/gaussian_2.png", width: 100%)
      #align(center)[(a) Kernel scale $sigma = 2$]
    ],
    [
      #image("/resources/img/gaussian_02.png", width: 100%)
      #align(center)[(b) Kernel scale $sigma = 0.2$]
    ],
  ),
  caption: [
    Matching of a square onto a circle using a Gaussian kernel.
     ],
)
 Such a kernel enforces the preservation of structures larger than the characteristic length $sigma$, by strongly correlating the motion of points $x, y$ such that $k(x, y) approx 1$.
    Note how the large scale (a) maintains the straight edges of the square, while the small scale (b) allows the points to break structure and fit the circle perfectly.

== Reduction of the Matching Problem to initial momentum

In this section, we leverage the Riemannian structure of the Landmarks shape space to interpret the Matching problem as a geodesic problem. Then, by exploiting the conservation of energy principle on geodesics, an idea originating from classical mechanics, we reduce the problem to the space of initial momentum $T_S^* cal(S) tilde.eq RR^(3 n)$.

=== Recap on the current state of the problem
We have established that our shape space is a Riemannian manifold equipped with a cometric $K_S$. We can now reinterpret our original optimization problem through this geometrical lens. We denote $S_0, S_1$ respectively our source and target shapes.
Recall that we are minimizing the energy functional over $L^2_(V  , S)$:
$ E(v) = lambda integral_0^1 norm(v_t)_V^2 dif t + d_cal(S)(Phi_1^v). $

Using our Riemannian formalism, the integral term is exactly the cost / length of the trajectory $S(t)$ on the manifold, for $t in [0,1]$:
$ L_(Phi^v_1)^(S_0) = integral_0^1 P(t)^T K_(S(t)) P(t) dif t. $

Where we recall that $ Phi_t^v (q_i (0)) = q_i (t)$, thus $ S(t) = (q_1(t) , dots , q_n (t))$ is the shape deformed after time $t$ by the diffeomorphism obtained by integrating the time-dependent vector field $v$ up to $t$.

=== Matching Problem as a Geodesic
In the following sections, $v$ is a minimizer of $E$, which maps $S_0$ to $S(1) = Phi_1^v (S_0) approx S_1$.  It is thus the optimal deformation of $S_0$ into $S(1)$. Consequently, it is also a minimizer of $L_(Phi^v_1)^(S_0)$.

We interpret this geometrically by the fact that $S(t)$ is a _length-minimizing_ curve on $cal(S)$ between $S_0$ and $S(1)$. By definition, this optimal deformation $S_0 --> S(1)$ is thus a *geodesic* of $cal(S)$.

As mentioned earlier, geodesics (defined as locally straight curves) are in general, not unique. Our optimal trajectory is a length-minimizing geodesic for the chosen formulation.


=== Hamiltonian Dynamics and Conservation of Energy

We have established that our optimal deformation is a geodesic. Earlier /* @sec:initial-velocity */, we have argued that, intuitively, given source and target, the only parameter that determines a geodesic and the time it takes is the initial *velocity* vector. 
This is nothing more than a *classical mechanics* argument. We can view the evolving shape as a particle moving through a curved space, governed by physical laws. The state of such a system is described by its *Total Energy* defined as the sum of Kinetic energy ($K_E$) and Potential energy ($P_E$):

$ "Total Energy (H)" = underbrace("Kinetic Energy" (K_E), "Motion / Inertia") + underbrace("Potential Energy" (P_E), "External Forces") . $

In our specific case, no external forces apply to the deformation hence the absence of potential energy.\
The shape thus behaves as a **free particle**: it "coasts" along the manifold solely due to its initial momentum, thus following a geodesic (equivalent to "not turning" in our previous walking analogy). A fundamental law of mechanics is the *Conservation of Total Energy* for isolated systems. Since $P_E=0$, this implies the *Conservation of Kinetic Energy*. 

The ubiquity of Riemannian geometry in the description of physical systems allows us to formalize this via a theorem true for every Riemannian manifold, originally emanating from the *Hamiltonian* formulation of classical mechanics.

We define the *Hamiltonian* for every shape $S$, momentum $P$ and associated velocity $v^P = K_S P in V$ as:
$ H(S, P) = 1/2 P^T K_S P = 1/2 norm(v^P)_V^2 . $


#theorem("Geodesic equations.")[
Any geodesic trajectory $(S(t), P(t))$ on our manifold satisfies the canonical system:

$ cases(
  (partial S) / (partial t) &= partial_P H(S, P) = K_S P,
  (partial P) / (partial t) &= -partial_S H(S, P) = -1/2 nabla_S (P^T K_S P)
) . $



]
We have already encountered the first equation: it is exactly the flow equation for the landmarks we derived /* @sec:landmark-flow */ using RKHS theory. 
This is not surprising: they are both interpretations of the same trajectory $S(t)$.
The second equation, however, is completely new to us: it tells us how the momentum (thus velocity) should evolve. It arises specifically when interpreting our optimal deformation as following a geodesic.

A simple application of the chain rule combined with the substitutions of both equation then yields the following result.


#theorem("Conservation of Energy.")[
Let $(S(t), P(t))$ be a solution to the canonical system. We have:
$ frac(dif, dif t) H(S(t), P(t)) = 0 => H(S(t), P(t)) = H(S_0, P_0) . $

The total energy is conserved.
]

Applying the theorem to our Hamiltonian containing only kinetic energy, we get:

$  integral_0^1 norm(v_t)_V^2 dif t = integral_0^1 2 H(S_0, P_0) dif t = 2 H(S_0, P_0) = P_0^T K_(S_0) P_0 . $


The entire trajectory $S(t)$ is fully determined by its initial momentum $P_0 = (p_1, ... , p_n )^T in (RR^3)^n$.

=== Matching Problem over initial momentum space


#definition("Matching Problem over momentum space.")[ Let $lambda > 0$.   The Matching problem over $L^2_(V,  S)$ .  $ op("Minimize") quad E(v) = lambda integral_0^1 norm(v_t)_V^2 dif t + d_cal(S) (Phi^v_1) . $
Is equivalent to the following problem over $RR^(3 n)$:

$ op("Minimize") quad E(P_0) = lambda P_0^T K_S P_0 + d_cal(S) (Phi_1^(v_(P_0))) . $

]


Starting from an infinite dimensional setting over $L^2_V$, choosing $V$ to be an RKHS space allowed us to reduce the matching problem to a finite dimensional problem over the time interval $[0,1]$. Finally, interpreting our problem as finding a geodesic in a Riemannian manifold reduced the problem to finding the optimal initial momentum $P_0$, which has the dimension of our shape.


Solving this inexact matching problem can be interpreted as approximating the $"Log"_(S_0)$ map at the source shape $S_0$, in the cotangent space.
=== The geodesic distance

We define the geodesic distance between shapes $S_0$ and $S_1$ as: $ d^2(S_0, S_1) = min_(P_0 in RR^(3 n )) E(P_0) $

Where the matching problem is taken with respect to shapes $S_0$ and $S_1$.
== Solving the Matching Problem

In this section, we describe how to computationally solve the Matching problem in its reduced form and present our numerical implementation using the open-source Python library scikit-shapes.


=== Computing the energy


Given a momentum $P_0$, we recall the expression of the matching term:
$ d_cal(S)(Phi_1^(v_(P_0))) = norm(Phi_1^(v_(P_0))(S_0) - S_1)_2^2 $ 

 Geometrically, we know our optimal deformation *must* follow a geodesic, thus $Phi_1^(v_(P_0)) (S_0)$ is obtained by *shooting* the shape along the geodesic defined by $P_0$. This is exactly computing the exponential map $op("Exp"_(S_0))$ at the source shape in the cotangent space. 

Numerically, this is done by integrating the time-discretized Hamiltonian system with $P_0$ as initial condition.
The scikit-shapes library implements an RK4 scheme for solving this system, with $n_("steps")$ as the discretization parameter.


=== Optimization 

Now that we know how to compute the energy $E(P_0)$, the last step is to minimize this cost function.
In scikit-shapes, this is achieved by leveraging the modern computational graph of *PyTorch*.

- *Automatic Differentiation:*
   Since the geodesic shooting is composed entirely of differentiable tensor operations, we do not need to derive or implement the complex adjoint equations manually. We simply leverage PyTorch's autograd engine to compute the exact gradient via backpropagation:
   $ nabla_(P_0) E = (partial E) / (partial P_0) $

- *The L-BFGS Optimizer:*
  Standard Gradient Descent is often too slow for deformation problems where the energy landscape can be ill-conditioned (narrow valleys). Instead, the standard choice in Computational Anatomy is the *L-BFGS* algorithm @younes2010shapes.
   
   This is a quasi-Newton method that approximates the inverse Hessian matrix of the energy using the history of past gradients. It offers the crucial advantage of avoiding the storage of the full $3n times 3n$ Hessian matrix, making it scalable to shapes with thousands of landmarks like femurs.

In general, the solver iteratively updates $P_0$ using this gradient information until the energy stabilizes or the gradient norm falls below a tolerance threshold.
In practice, we lack information about the energy landscape in deformation problems, and the problem is highly dimensional ($3 times 18291 = 54873$ for our femur data). We thus set a fixed number of iterations $n_("it")$ for the descent.

== Statistics on the Shape Manifold

The linear structure of the cotangent space $T_S cal(S) tilde.eq RR^(3 n)$ is the great reward for our efforts of characterization of the optimal shape deformations: one initial condition, a momentum vector, which is linearly stable: the sum of two valid momentum vectors is another valid momentum vector, tangent to the manifold. This all arises from the smooth manifold assumption of the shape space.

Beware, however, that previous reasoning @local-bijection tells us the results of our method are only valid for "small enough" deformations of the atlas. However, probabilistic arguments tell us it is not a limitation to worry about, see @lytchak2025zeroprobabilitycutlocus. 
=== Atlas Construction: Computing the Fréchet Mean

In Computational Anatomy, an **Atlas** is a statistical model of the population structure. It consists of two main components:
1.  A **Template Shape** (or Mean Shape) $macron(S)$.
2.  A set of **Deformations** mapping this template to each individual subject $S_i$ in the dataset.

Mathematically, constructing an atlas corresponds to computing the **Fréchet Mean** (or Riemannian barycenter) of the population. Just as the arithmetic mean $S_a$ minimizes the sum of squared Euclidean distances, the Fréchet mean $macron(S)$ minimizes the sum of squared geodesic distances. 

$ macron(S) = arg min_S sum_(i=1)^K d^2 (S, S_i) . $

We provide a data-driven algorithm that computes the mean and the initial momentum $P_i$ necessary to reconstruct each shape from the mean by following a geodesic.

#block(width: 100%, stroke: 1pt + black, inset: 12pt, radius: 4pt, fill: luma(250))[
 *Algorithm: Atlas Construction via Geodesic Shooting*
  
  *Input:* 
  - Target Dataset ${S_1, ..., S_K}$
  - Kernel scale $sigma$, Regularization weight $lambda$
  - Step size $epsilon$
  - Optimizer (e.g., L-BFGS)
  
  *Initialization:* 
  - $macron(S) arrow.l$ Euclidean Mean of ${S_i}$  
  - ${P_0^i} arrow.l {0, ..., 0}$ 

  *Optimization Loop:* *While* $i < n_("atlas") $ *do*:
    
  1. *Registration Step (Log-Map approximation):*
       For each subject $S_i$, find the momentum $P_i$ that shoots the current atlas onto the subject:
       $ P^i_0 arrow.l "argmin"_(P_0^i) ( E(P_0^i)) $
       
  2. Compute the average momentum vector:
       $ macron(P) arrow.l frac(1, N) sum_(i=1)^N P_i $
       
  3. *Geodesic Update (Exponential Map):*
  
       Update the atlas by shooting it along the average momentum  for step size $epsilon$:
       $ macron(S) arrow.l Phi_1^(epsilon dot v_(macron(P))) (macron(S)) $
  *Returns:* 
  - Learned Template $macron(S)$
  - Subject-specific deformations of the template parameterized by ${P_0^i}$
]


One important aspect of this algorithm is that it relies on computing in the cotangent space to take advantage of the linear structure for interpretable operations. In order to do this, we need to go back and forth between $cal(S)$ and $T^*_macron(S) cal(S)$ via $op("Exp")$ and approximate-$op("Log")$ #footnote[Actually via their dual maps, but we aim to not make the presentation more confusing than it already is for the non-initiated reader.].


If the algorithm converges, the atlas is centered, i.e the mean momentum $macron(P_0)$ of the set of optimal momenta ${P_0^i}$ is $0$, which is to be expected intuitively.

=== Principal Geodesic Analysis: PCA in the (co)tangent space

Once the atlas is built, we can perform classical statistical analysis on the momentum space, isomorphic #footnote[Actually locally isomorphic. ] to the space of optimal deformations. We chose a linear PCA implementation.

This process, known as *Tangent PCA* or Principal Geodesic Analysis (PGA), identifies the principal modes of anatomical variation by diagonalizing the covariance matrix of the momenta. Let ${(lambda_k, U_k)}_k$ be the resulting eigenvalues (variances) and eigenvectors (principal directions of deformation).

We can now model any plausible deformation in the population as a linear combination of these principal modes. A new momentum vector $P(alpha)$ is constructed as:

$ P(alpha) = macron(P) + sum_(k=1)^M alpha_k sqrt(lambda_k) U_k $

where $macron(P)$ is the mean momentum (usually $approx 0$) and $alpha = (alpha_1, dots, alpha_M)$ are the coordinates in the reduced latent space (standard deviation units).

*Generative Model:*
To visualize the shape corresponding to a specific configuration $alpha$, we simply "shoot" the template along this synthesized momentum vector using the Riemannian exponential map:

$ S_("gen") = "Exp"_(macron(S)) ( P(alpha) ) = Phi_1^(v_(P(alpha))) (macron(S)) $

It is important to stress that while the PCA is linear, it is done in the cotangent space hence doesn't assume the data is located on a linear subspace of the ambient space. Via geodesic shooting, it allows us to explore the shape manifold by moving along geodesic sub-manifolds defined by the principal components, guaranteeing that the generated shapes remain diffeomorphic to the template.

== Results

The LDDMM method as detailed in this report is very computationally intensive. At the time of writing (02/02/26), the atlas for a high precision configuration of the method is being built. The parameters are still being finalized.

We will update this section once the definitive results are in.
