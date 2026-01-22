#import "@preview/touying:0.5.5": *
#import "@preview/clean-math-presentation:0.1.1": *

#show: clean-math-presentation-theme.with(
  config-info(
    title: [Modeling Project: 3D Reconstruction of Femurs],
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

= Introduction

#slide(title: "Context")[
  - Clustering
  - Shape analysis
  - ... Malik TODO
]



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



= PCA

#slide(title: "PCA")[
  Quick explanation of PCA
]

= Linear algebra implementation

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
  == Neural network architecture
  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      - Input layer
      - Hidden layers
      - Output layer
      - Forward propagation
    ],
    [
      #image("../fig/NN_Diagram.png", width: 100%)
    ]
  )

  == Autoencoder structure
  - Input layer: $54876$ neurons
  - Encoder: series of fully connected layers reducing dimensionality
  - Latent space: compressed representation of input data we choose size 10.
  - Decoder: series of fully connected layers reconstructing the original data
  - Output layer: reconstructed 3D point cloud

]


= Training process

#slide(title: "Training process")[
  - pb of sigmoid fonction : vanishing gradient
  - enregistrer le RDN dans un binaire au lieu d'un txt pour gagner en performance et place
  - normalisation/standardisation des données d'entrée et de sortie
  - differentes fonctions d'activation
  - cost function : MSE because of point correspondance
  - non linear last layer
  
  This is the training process slide.
  == Backpropagation algorithm

  == Choice of loss function
]

= Optimization techniques

== Multithreading
#slide(title: "Multithreading")[

#speaker-note()[
  NN is quiet slow to train --> multi-threading system to speed up the process. \
  
  1. create a thread for each operation that can be parallelized. But the overhead created by the creation of threads is too important because of the large number of threads created compared to the time saved by parallelizing the operation. \
  2. split the operations in a fixed number of threads (depended of the computer threads, basically 4 or 8). Each thread will compute a part of the result vector.
    ]

  === Motivation
    - Speed up training process
    - Efficiently utilize multi-core processors
    - For Matrix $times$ Vector multiplication

  === `std::thread` approaches
    1. For every parallelizable operation, create a thread
    2. Create a fixed number of threads at the beginning
    3. Thread pool
  
  === OpenMP
    - Simple to implement with compiler directives
    - Automatically manages thread creation and workload distribution

  #speaker-note()[  3. thread pool: create a fixed number of threads at the beginning of the program.  This approach reduces the overhead of thread creation and destruction, leading to better performance. We didn't implement beacause of a lack of time. \

  OpenMP expliquer brievement le principe et a quoi ca sert]
#figure(
  image("../fig/perf_multithreading.png", width: 80%),
  caption: [
    Performance comparison between single-threaded and multi-threaded training (with different values of the treshold parameter) // we don't see the caption in touying (here)
  ],
) <perf_multithreading>

]

== Memory allocation
#slide(title: "Memory allocation")[
  Cache allocation

  Performance improvement
]

= Visualization

#slide(title: "First visualizations")[
  == `visuFemur.py`
  - Quick OBJ mesh viewer
  - To debug and visualize femur meshes generated by the RDN

  == `compare_femur.py`
  - Compare two femurs meshes (between original and reconstructed femurs for example)

  == `latent_explorer.py`
  - Interactive latent space explorer (sliders)
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

= Live Demo

#focus-slide()[
  Live demo
]

#slide(title: "Conclusion and ameliorations")[
  - RDN with bigger train dataset (test with data augmentation of linear PCA (may not add anything since it's linear))
  - Thread pool

]

#show: appendix

= References

#slide(title: "References")[
  #bibliography("../bibliography.bib", title: none)
]
