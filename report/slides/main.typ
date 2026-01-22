#import "@preview/touying:0.5.5": *
#import "@preview/clean-math-presentation:0.1.1": *

#show: clean-math-presentation-theme.with(
  config-info(
    title: [Statistical and Neural Approaches to 3D Femur Modeling],
    authors: (
      (name: "Boyer Timothé"),
      (name: "Hacini Malik"),
      (name: "Lainé Martin"),
      (name: "Mouret Basile"),
    ),
    date: datetime(year: 2026, month: 01, day:23),
  ),
  config-common(
    slide-level: 3,
    //handout: true,
    //show-notes-on-second-screen: right,
  ),
  progress-bar: true,
)

#title-slide(
  // On met tout dans logo1 avec une grille
  logo1: block(width: 50%)[
    #grid(
      columns: (auto, 1fr, auto), // Gauche - Espace élastique - Droite
      align: horizon,             // Aligne verticalement les images
      image("../assets/im2ag.png", height: 2.5cm),
      [],                         // Case vide qui pousse les images aux bords
      image("../assets/ensimag.png", height: 2.5cm)
    )
  ],
  logo2: none // On s'assure que le 2ème est vide
)
== Outline <touying:hidden>

//#components.adaptive-columns(outline(title: none))



= Data
#slide(title: "Data")[
  - 24 scans of femurs
  - 12 left and 12 right
  - 18097 3D points corresponding to 54292 parameters

  #v(2em)
  #figure(
    grid(
      columns: (1fr, 1fr),
      gutter: 1em,
      [
        #image("../fig/femur_3D.png", width: 70%)
      ],
      [
        #image("../fig/femur_3D_edges.png", width: 66%)
      ]
    ),
    
    caption: [Example of a femur mesh],
  ) <femur_pointcloud>
]



= Linear PCA

#slide(title: "Linear PCA : Principle")[
  #figure(
  grid(
    columns: (auto, auto, auto, auto),
    align: (center + horizon),  
    gutter: 1em, // Space between images
    image("../placeholder/placeholder_pca.png", width: 110%),
    pause,
    $-->_("Ellipsoid fit")$,
    image("../placeholder/placeholder_pca.png", width: 110%),
  ),
 // Optional: remove if no caption needed
)
]

#slide(title: "Linear PCA : Principle")[
 #figure(
  grid(
    columns: (auto, auto, auto, auto),
    align: (center + horizon),  
    gutter: 1em, // Space between images
    image("../placeholder/placeholder_pca.png", width: 110%),
    pause,
    $-->_("Eigendecomposition")$,
    image("../placeholder/placeholder_pca.png", width: 110%),
  ),
)
]

#slide(title: "Linear PCA : A subtle potential issue")[
- PCA assumes the data is distributed as a gaussian point cloud
- If the data is made up of distinct clusters (e.g *healthy* and *unhealthy* femurs), the method breaks down.
- Luckily for us, the femur data originates from all types of individuals which averages out the cluster effect.

  #figure(
    image("../placeholder/placeholder_pca.png", width: auto, height: 50%),
    caption: [PCA is unadapted to datasets with highly clustered structure],
  ) <fimg-label>
]
#slide(title: "Linear PCA : Foundation")[
  // Slide 1: Mathematical Foundation
// Context: We treat each femur as a vector of landmark coordinates.


#v(1em)
We represent each femur as a vector $S_i in RR^(3P)$.


We compute the *Mean Femur* $macron(S)$ and the sample covariance matrix $C$:
$ macron(S) = 1/N sum_(i=1)^N S_i quad , quad S = 1/(N-1) sum_(i=1)^N (S_i - macron(S)) (S_i - macron(S))^top $

  #figure(
    image("../fig/femur_3D.png", height: 40%),
    caption: [The (arithmetic) mean femur], 
  ) <fimg-label>
]

#slide(title: "Linear PCA : Eigendecomposition")[
  // Slide 2: Eigen decomposition and Principal Components
PCA orthodiagonalizes the covariance matrix to find the principal directions.
- The *eigenvectors* $v_k$ are the *Principal Components* (directions of variance).
- The *eigenvalues* $lambda_k$ represent the variance captured by component $k$.
- Taking the $p$ top components  effectively reduces the dimensionality of the dataset to $p$.  #figure(
    image("../fig/pca_illustration.png", width: auto, height: 50%),
    caption: [PCA reduces the data to it's $k$ most meaningful components],
  ) <fimg-label>


]

#slide(title: "Linear PCA Interpretation : Modes of Variation")[


Any femur instance $S$ in the dataset can be approximated as the mean shape plus a weighted sum of the principal components:

$ S approx macron(S) + sum_(k=1)^K omega_k v_k $

- $omega_k$: The standard deviation specific to this individual w.r.t component $k$.

 We can then visualize all possible linear deformations and try to interpret them clinically !


]

#slide(title: "Linear PCA : Results and Limitations")[
PCA assumes that the shape space is flat (a linear subspace of $RR^(3N)$).
- No reason to believe all linear transformations are anatomically plausible
-  True anatomical deformations can be non-linear (e.g. torsion or bending).

#figure(
  image("../fig/sphere_first.png", width: auto, height: 60%),
  caption: [Eculidean transformations of points do not preserve underlying structure],
) <fimg-label>
]
= Linear algebra implementation



= Neural network

#slide(title: "Neural network")[
  == Neuron structure
  Given an input $x in bb(R)^n$

  The characteristics of a neuron are:  
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      - Weights vector $w = (w_1, w_2, dots , w_n)$
      - Bias term $b$
      - Weighted sum function $f(x) = w . x + b$
      - Activation function $Phi$
    ],
    [
      #align(center)[
      #image("../fig/Neuron_diagram.png", width: 80%)
      ]
    ]
  )
  
  #v(1em)

  *Output of a neuron:*

  $ (Phi compose f)(x) = Phi(f(x)) $
]

#slide(title: "Neural network")[
 


#box(width: 100%)[
  #grid(
    columns: (auto, auto, 19em, auto, auto),
    align: (center + horizon),

    // 1. Original (Smaller)
    stack(dir: ttb, spacing: 0.5em,
      image("../fig/original_L_Femur_11.png", height:70%),
      [Original Femur]
    ),

    // 2. Preprocessing
    stack(dir: ttb, spacing: 0.5em,
      $arrow.long$, 
      text(size: 0.8em)[Preprocessing]
    ),

    // 3. Network (Big)
    image("../fig/autoencoder.svg", width: 19em),

    // 4. Postprocessing
    stack(dir: ttb, spacing: 0.5em,
      $arrow.long$,
      text(size: 0.8em)[Postprocessing]
    ),

    // 5. Reconstructed (Smaller)
    stack(dir: ttb, spacing: 0.5em,
      image("../fig/reconstructed_L_Femur_11.png", height:65%),
      [Reconstructed Femur]
    )
  )
]
  
]



#slide(title: "Training process")[

  = First Model
    - #underline[*Layers*]: {54873, 1024, 256, 32, *10*, 32, 256, 1024, 54873}
    - #underline[*Activation function*] : Sigmoid
    - #underline[*Loss function*] : MSE
    - #underline[*Preprocessing*] : MinMax Normalization for each coordinate
    - #underline[*Training*] : 1000 epochs
  #pause
  = Problems
    - Slow to train (> 100M parameters $approx$ 500MB in memory)
    - Vanishing gradient
    - Loss distances aren't proportional to femur distance
    - Boxed output
]

#slide(title:"Training Process")[
  = Solutions
    - Use a *linear output layer* so the model can take every value
    - Change the activation function for *Tanh* and *LeakyReLU*
    - Reduce the layer sizes
    - Preprocessing : remove the *mean femur* and normalizing all coordinates *equally*
    #pause
  = Latest Models
    - *Layers*: {54873, 1024, 256, 32, *10*, 32, 256, 1024, 54873}
    - *better activation functions*: tanh and LeakyReLU
    - *Linear* last layer
    - New *Preprocessing*
    - Longer *training*: 5000 epochs
]

#slide(title: "Linear algebra implementation")[
  == Custom library design
  - Built on top of *Eigen* for efficient internal storage
  - Template-based classes: `Vector<T>`, `Matrix2D<T>`, `Matrix2DSquare<T>`
  - Supports multiple numeric types: `float`, `double`, `int`, `long`, etc.

  == Main classes
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      *Vector<T>*
      - Scalar/dot product
      - Hadamard product
      - Outer product
    ],
    [
      *Matrix2D<T>*
      - Matrix multiplication
      - Transpose
      - Row/column extraction
    ]
  )
]

= Optimization techniques

== Multithreading

===  Motivation of multithreading
#slide(title: "Motivation of multithreading")[
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
    - Speed up the training process by using *multi-core processors*.
    #align(center)[
      #image("../fig/threads.png", width: 50%)
    ]
    ],

    [
      #image("../fig/multi_vs_single_threading.jpeg", width: 100%)
      #pause
    ]
  )
    #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #v(2em)
    - For *Matrix $times$ Vector* multiplication :
    ],
    [
  $ mat(
        w_11, w_22, dots, w_(1n);
        w_21, w_22, dots, w_(2n);
        dots, dots, dots, dots;
        w_(m 1), w_(m 2), dots, w_(m n)
      )
      times mat(x_1; x_2; dots; x_n)
      = mat(y_1; y_2; dots; y_m) $
    ])
]
  
#slide(title: [Multithreading with `std::thread`])[
1. For *every* parallelizable operation, create a thread
#pause
2. Create a *fixed* number of threads at the beginning of the function
#image("../fig/btop2_opt.gif")

#pause
    #grid(
    columns: (1fr, 1fr, 1fr, 1fr),
    [
3. *Thread pool*],[
#image("../fig/Thread_pool.svg.png", width: 155%)
])

]
#slide(title: "OpenMP Multithreading")[
  - *Automatically* manages thread creation and workload distribution
  - *Simple to implement* (near to sequential code)

```cpp
omp_set_num_threads(omp_get_max_threads());
#pragma omp parallel for

for (size_t j = 0; j < m_rows; ++j) {
    T sum = 0;
    for (size_t i = 0; i < m_cols; ++i) {
        sum += m_data(j, i) * vec(i);
    }
    result(j) = sum;
}
return result;
```



]
#slide(title: "Performance Graph")[
#align(center)[
  #image("../fig/perf_multithreading.png", width: 82%)
  - Treshold parameter = number of parameters in the weitght matrix above which we use multithreading.

]

]

== Memory allocation
#slide(title: "Memory allocation")[
  == Data Oriented Design
  #grid(
    columns: (auto, auto),
    [
   - improve *cache locality* by switching rows and columns acces (*$times 2$ speedup*)
   - *Preallocation* of variables and Memory optimized functions (*$times 4$ speedup*)
- In total $==>$  *$times 8$ speedup*, going from 58 seconds to 7 seconds per epoch
],
    image("../fig/memory_speed.png", height:90%),
    

  )



]

= First Visualization

#slide(title: "First visualizations")[
  #grid(
    columns: (1fr, 1fr, 1fr),
    gutter: 1em,
    [
  == `visuFemur.py`
  #image("../fig/bone_visu.png", width: 100%)
  #pause
],[
  == `compare_femur.py`

  #image("../fig/heat_map.png", width: 100%)
  #pause
],[
  == `latent_explorer.py`
#image("../fig/10sliders.jpeg", width: 150%)
]
  )
]


= PCA on the latent space

#slide(title: "PCA on the latent space")[
  Using the latent space we plot the data in 3D using only 3 components.

#v(1em)

#figure(
  image("../fig/latent_space_plane_comparison.pdf", width: 60%),
  caption: [Latent space visualization in 3D
  
  We remark that the data seems organized in a plane.
  ]
)
]


#slide(title: "PCA on the latent space")[
  #figure(
    grid(
      columns: (1fr, 1fr),
      gutter: 1em,
      [
        #image("../fig/pca_cumulative_variance_NeuralNetwork_centered_tanh_5000.png", width: 100%)
      ],
      [
        #image("../fig/pca_cumulative_variance_NeuralNetwork_centered_LReLU.png", width: 100%)
      ]
    ),
    caption: [Cumulative variance with PCA on the latent space.]
  )
]

= Results and limitations
#slide(title: "Results and limitations")[
== Results
  - Neural Network captured *components* of the dataset
  - It can be used to *generate* visually plausible femurs
  - PCA on the latent space shows that the data is organized in a *subspace* of lower dimension
#pause

== Future work
  - *Augment* dataset using PCA
  - Increase Neural Network *depth*
  - Using *threadpools*
  - Train on *unhealthy femurs*
  - Implement the *Variational Autoencoder* architecture
]

= LDDMM : Riemannian geometry to the rescue

#slide(title: "Principles of LDDMMM")[

  *The Core Idea*
Instead of treating femurs as vectors in a flat space (Linear PCA), we treat them as points on a curved, nonlinear *Riemannian manifold*.

*Why?*
- *Linear operations* (adding shapes) can break anatomy (self-intersections).
- *Diffeomorphisms* (smooth, invertible deformations) preserve topology.

*The Framework*
We analyze the "deformation" required to morph a source shape $S$ into a target $T$.
- The "size" of this deformation is measured by a geodesic distance.
- Statistics are computed on these deformations, not on the point coordinates directly.

]

#slide(title:"LDDMM : Equipping the shape space with a Riemannian metric")[
We define a distance between shapes based on the *energy* required to deform one into the other.

*Diffeomorphisms Group*
We consider the group of diffeomorphisms $phi in "Diff" Omega)$

The cost of a deformation is determined by the *velocity field* $v_t$ that generates it. We define a norm on velocity fields using a differential operator $L$ (enforcing smoothness):

$ norm(V)^2 = integral_Omega |L v(x)|^2 d x $

The higher the energy, the larger the deformation.

]



#slide(title:"LDDMM Algorithm" )[
*1. Flow Equation (Generating Deformations)*
A deformation $phi_1$ is generated by integrating a time-dependent velocity field $v_t$ over $t \in [0, 1]$:
$ d phi/ ( d t) = v_t compose phi_t $

*2. Energy Minimization (Geodesic Shooting)*
We find the optimal flow $v_t$ that minimizes:
It is uniquely determined by its **Initial Momentum** $m_0$.

]

#slide(title: "LDDMM for SSI")[
  *1. Atlas Building (The Mean)*
We compute the "Mean Femur" (Atlas $macron(S)$) as the shape that minimizes the sum of squared geodesic distances to all subjects in the population (Fréchet Mean).

*2. Tangent Space PCA*
Since the manifold is curved, we cannot do PCA directly on shapes.
- We map all subjects $S_i$ to the *Tangent Space* of the Atlas (via the initial momenta $m_0^i$).
- The Tangent Space is a vector space (linear).
- We perform *PCA on the momenta*

*Output:* Principal Geodesic Analysis (PGA). The "modes" are trajectories of deformation applied to the mean shape.

]

#slide(title: "Comparison with Linear PCA")[

#table(
  columns: (1fr, 1fr, 1fr),
  inset: 10pt,
  align: horizon,
  [*Feature*], [*Linear PCA*], [*LDDMM (Tangent PCA)*],
  [**Geometry**], [Flat (Euclidean)], [Curved (Riemannian)],
  [**Deformation**], [Displacement vectors], [Diffeomorphic Flow],
  [**Topology** preservation], [Not guaranteed (folding risk)], [Preserved (anatomical)],
  [**Interpolation**], [Straight lines], [Geodesic curves],
  [**Cost**], [Fast], [Computationally intensive]
)
]


#slide[
  // Lien canva: https://www.canva.com/design/DAG_JSW4-ug/irIZxhoOdvJ4ilOG2dz6YQ/edit
  #place(
    top + left,
    dx: -2cm,
    dy: -2.5cm,
    image("../fig/last_slide.gif", width: 125%, height: 125%)
  )
]
