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

  == Goals of the Project
  - Implement a autoencoder for 3D reconstruction of femurs
]

= Linear Algebra Implementation

#slide(title: "Linear Algebra Implementation")[
  == Custom Library Design
  - Built on top of *Eigen* for efficient internal storage
  - Template-based classes: `Vector<T>`, `Matrix2D<T>`, `Matrix2DSquare<T>`
  - Supports multiple numeric types: `float`, `double`, `int`, `long`, etc.

  == Main Classes
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

= Neural Network Architecture

#slide(title: "Neural Network Architecture")[
  This is the neural network architecture slide.
]


= Training Process

#slide(title: "Training Process")[
  This is the training process slide.
]

= Optimization Techniques

#slide(title: "Optimization Techniques")[
  == Multithreading

  == OpenMP

  == Comprarison with Benchmarks
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
