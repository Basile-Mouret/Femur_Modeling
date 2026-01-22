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

== Context
#slide(title: "Context")[
  - Clustering
  - Shape analysis
  - ... Malik TODO
]

== Data description
#slide(title: "Data description")[
  - use the `visuFemur.py` to show some femur meshes: the femur, the grid (triangle mesh), ... (there is different parameters in the script to show or hide some elements)
  - how we obtained the data ?
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
      - Bias term
      - Weighted sum function $f(x) = w . x + b$
      - Activation function $Phi$
    ],
    [
      PLACEHOLDER for Neuron Diagram
    ]
  )
  
  #v(2em)

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
      PLACEHOLDER for Neural Network Diagram
    ]
  )

  == Autoencoder structure


#box(width: 100%)[
  #grid(
    columns: (auto, auto, auto, auto, auto),
    align: (center + horizon),

    // 1. Original (Smaller)
    stack(dir: ttb, spacing: 0.5em,
      image("../fig/original_L_Femur_11.png", height: 3.5cm),
      [Original Femur]
    ),

    // 2. Preprocessing
    stack(dir: ttb, spacing: 0.5em,
      $arrow.long$, 
      text(size: 0.8em)[Preprocessing]
    ),

    // 3. Network (Big)
    stack(dir: ttb, spacing: 0.5em,
      // Increased height to 7cm to make it dominant
      image("../fig/autoencoder.svg", height: 6cm),
      [Neural Network]
    ),

    // 4. Postprocessing
    stack(dir: ttb, spacing: 0.5em,
      $arrow.long$,
      text(size: 0.8em)[Postprocessing]
    ),

    // 5. Reconstructed (Smaller)
    stack(dir: ttb, spacing: 0.5em,
      image("../fig/reconstructed_L_Femur_11.png", height: 3.5cm),
      [Reconstructed Femur]
    )
  )
]
  
]



#slide(title: "Training process")[

  = First Model
    - Layers: {54873, 1024, 256, 32, 10, 32, 256, 1024, 54873}
    - Activation function : Sigmoid
    - Loss function : MSE
    - Preprocessing : MinMax Normalization for each coordinate
    - Training : 1000 epochs
  = Problems
    - Slow to train
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
  = Latest Models
    - Layers: {54873, 512, 64, 10, 64, 512, 54873}
    - Better activation functions: tanh and LeakyReLU
    - Linear last layer
    - New Preprocessing
    - Longer training: 5000 epochs
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
  = Memory Bandwidth Bottleneck
   - improve cache locality by switching rows and columns acces (x2 speedup)
   - Preallocation of variables and Memory optimized functions (MultiplyTranspose) (x4 speedup)

Total :  *8x speedup*, going from 56 seconds to 7 seconds per epoch

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

  #grid(
    columns: (1fr, 1fr),
    gutter: 1em,
    [
      #image("../fig/pca_cumulative_variance_NeuralNetwork_centered_tanh_5000.png", width: 100%)
    ],
    [
      #image("../fig/pca_cumulative_variance_NeuralNetwork_centered_LReLU.png", width: 100%)
    ]
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
