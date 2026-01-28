// results achieved with the chosen method + performance assessment (computation time, accuracy, code quality, ...).

== Neural Network results

We decide to plot the data in the latent space using only these three components. To see if there is some structure in the data.

#v(1em)
#figure(
  image("../../fig/latent_space_plane_comparison.pdf", width: 70%),
  caption: [Latent space representation using the first three principal components. We draw one plane approximating the data distribution.]
)
#v(1em)

We can see that the data is quite well approximated by a plane in this 3D space. That allow us to add details on the visualization of the latent space.

Combinating these tswo observation that means that we can play with two parameters to explore the latent space.


The goal of the encoder is to reduce the dimensionality of the femur shapes from 54873 to 10. 

Even with this strong reduction, it's still complicated to visualize the results. We decide do a PCA on the latent space to see if we can focuse on fewer dimensions.

#v(1em)
#figure(
  image("../../fig/pca_cumulative_variance_NeuralNetwork_centered_tanh_5000.png", width: 70%),
  caption: [Cumulative variance explained by PCA on the latent space]
)
#v(1em)

We can see that the first three principal components explain almost 85% of the variance in the latent space.



