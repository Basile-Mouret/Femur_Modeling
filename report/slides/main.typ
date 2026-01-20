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
  progress-bar: false,
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

#components.adaptive-columns(outline(title: none))

= Introduction

#slide(title: "Introduction")[
  This is the introduction slide.

  == Goals of the project
  - Implement a autoencoder for 3D reconstruction of femurs
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

= Neural network architecture

#slide(title: "Neural network architecture")[
  This is the neural network architecture slide.
]


= Training process

#slide(title: "Training process")[
  This is the training process slide.
]

= Optimization techniques

#slide(title: "Optimization techniques")[
  == Multithreading

  == OpenMP

  == Comparison with benchmarks

  == Memory allocation
]


= Live Demo

#focus-slide()[
  Live demo
]

#show: appendix

= References

#slide(title: "References")[
  #bibliography("../bibliography.bib", title: none)
]
