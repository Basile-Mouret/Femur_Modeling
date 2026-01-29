// results achieved with the chosen method + performance assessment (computation time, accuracy, code quality, ...).

== Encoder results 

The goal of the encoder is to reduce the dimensionality of the femur shapes from 54873 to 10. 

We decide to plot the data in the latent space using only these three components. To see if there is some structure in the data.

#figure(
  image("/resources/img/latent_space_plane_comparison.pdf", width: 70%),
  caption: [Latent space representation using the first three principal components. We draw one plane approximating the data distribution.]
)

With this plot, we can see that the data is not uniformly distributed in the latent space. We decide do a PCA on the latent space to see if we can focuse on fewer dimensions.

#v(1em)
#figure(
  image("/resources/img/pca_cumulative_variance_NeuralNetwork_centered_tanh_5000.png", width: 70%),
  caption: [Cumulative variance explained by PCA on the latent space]
)
#v(1em)

With this plot, we can see that the first six principal components explain almost 90% of the variance in the latent space. This means that we can reduce the dimensionality of the latent space from 10 to 6 without losing much information.

