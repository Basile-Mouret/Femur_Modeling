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

= Dimensionality reduction
#slide(title: "What is there to learn ?")[
  #grid(
  columns: (auto,auto),
  [
  - _A priori_, femurs are _random_ vectors of $RR^(3N)$
  - In practice, femurs have structure : not any random $RR^(3N)$ vector represents an anatomically plausible femur !
    - Compact, distinct shaft and heads...
    #v(1em)
  - Femur data thus lies in a low dimensional structure
    - Goal is learning this structure $-->$ need for dimensionality reduction methods
  ],
  
  figure(
    image("../fig/wemby.jpeg", width: auto, height: auto),
    caption: [Victor Wembanyama],
  ) 
  )<fimg-label>
]


= Linear PCA

#slide(title: "Linear PCA : Principle")[
  #figure(
  grid(
    columns: (auto, auto, auto, auto),
    align: (center + horizon),  
    gutter: 1em, // Space between images
    image("../fig/pca_start.gif", width: 110%),
    pause,
    $-->_("Ellipsoid fit")$,
    image("../fig/pca_ellipse.gif", width: 110%),
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
    image("../fig/pca_ellipse.gif", width: 110%),
    pause,
    $-->_("Eigendecomposition")$,
    image("../fig/pca_full.gif", width: 110%),
  ),
)
]

#slide(title: "Linear PCA : A subtle potential issue")[
- PCA assumes the data is distributed as a gaussian point cloud
- If the data is made up of distinct clusters (e.g *healthy* and *unhealthy* femurs), the method breaks down.
- Luckily for us, the femur data originates from all types of individuals which averages out the cluster effect.

  #figure(
    image("../fig/pca_cluster.gif", width: auto, height: 50%),
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
  - Template-based classes
  - Supports multiple numeric types: `float`, `double`, `int`, `long`, etc.

  #pause 

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
1. For *every* parallelizable operation, create a thread \
  --> naive implementation : time spent creating and destroying threads is *larger* than the time saved by parallelization !
#pause
2. Create a *fixed* number of threads at the beginning of the function \
  --> better, but still a lot of thread creation/destruction !
#image("../fig/btop2_opt.gif")

]
#slide(title: [Multithreading with `std::thread`])[
3. *Thread pool*
#image("../fig/Thread_pool.svg.png", width: 80%)
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

#slide(title: "LDDMM : Big Picture")[

- Instead of treating femurs as vectors in a flat space, we would like to capture the finer underlying structure of the shape space, in order to understand anatomically plausible transformations
- LDDMM learns a representation of the data as a *curved* *Riemannian manifold* $cal(M)$
- Anatomical deformations are interpreted as following *geodesic* paths on the manifold

#figure(
  image("../fig/sphere_geodesic.png", width: auto, height: 50%),
  caption: [Manifold structure allows for "natural" statistics],
) <fimg-label>
]



#slide(title:"LDDMM : Big Picture")[

- The space of diffeomorphisms (smooth transformations) over $cal(M)$ is infinite dimensional.

- Based on the *Principle of least action* often respected in nature, we are only interested in the *cheapest* deformations w.r.t some *energy*.
- We define the geodesic distance by minimizing a *variational energy* (i.e cost) :
  $ E(v) = underbrace(
  1 / (2 sigma_R^2) integral_0^1 norm(  v_t )_V^2 dif t,
  "Regularity"
) + underbrace(
  1 / (2 sigma_M^2) sum_(i)^() abs(T_i - phi(S_i))  ,
  "Matching"
) $
- Computationally expensive $-->$ need for efficient non-convex optimization algorithms
]


#slide(title: [LDDMM : Geodesic statistics])[
- Statistics are computed w.r.t geodesic distance, not euclidean. 
  - Guarantee of staying on the manifold and being interpretable as plausible femurs !

- We compute the mean femur (Atlas $macron(S)$) with respect to the geodesic distance :
  $ macron(S) = arg min_S sum_(j=1)^N d_G (S, S_j)^2 $

#figure(
  image("../fig/sphere_geodesic.png", width: auto, height: 45%),
  caption: [Manifold structure allows for "natural" statistics],
) <fimg-label>

]

#slide(title:[LDDMM : From curved manifold to linear statistics] )[
- * Goal * : Main modes of variation of geodesic deformations.
- *Problem *: The space of geodesic deformations $cal(G)$ is curved  so standard linear PCA cannot be applied directly.
#pause
- The *geodesic shooting theorem* creates a 1-to-1 map between geodesic deformations of the mean shape and their *initial momenta*
   - This flattens $cal(G)$ into a finite-dimensional vector space $T_(macron(S)) cal(G) tilde.eq RR^(3N)$


#figure(
  image("../fig/geodesic_shooting.png", width: auto, height: 45%),
  caption: [Geodesic shooting linearizes the space of geodesic deformations],
) <fimg-label>
]


#slide(title: "LDDMM : Tangent PCA results and limitations ")[
- This linearization allows us to rigorously apply PCA on $T_(macron(S)) cal(G)$.
 - Main modes of variation around the mean femur $macron(S)$
- Deformations are not linear but geodesic
- Accounts for finer *local* deformation compared to Linear PCA which focuses on *global* variation.
- Requires less data than autoencoder, while being more interpretable

  #pause
* Limitations :*
  - Very computationally expensive to build the model

* Future work :*
  
  - Recent papers build *diffeomorphic autoencoders* to construct atlases $-->$ more efficient.
]

#slide[
  // Lien canva: https://www.canva.com/design/DAG_JSW4-ug/irIZxhoOdvJ4ilOG2dz6YQ/edit
  #place(
    top + left,
    dx: -2cm,
    dy: -2.5cm,
    image("../fig/last_slide.png", width: 125%, height: 125%)
  )
]
