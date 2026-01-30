== Linear Principal Component Analysis (PCA)

Out first approach to understand the shape variability of femur bones is through Linear Principal Component Analysis (PCA). 

=== Principle 

It's clear that in general, femur bones have a similar structure, but they can vary in size, curvature, and other shape characteristics. PCA is a statistical technique that helps us to identify and quantify these variations. It's a change of basis that aims to find the directions (principal components) in which the data varies the most. By projecting the data onto these principal components, we can reduce the dimensionality of the dataset while retaining most of the variance.

#figure(
  image("/resources/img/bone_visu.png", width: 20%),
  caption: [3D visualisation of a femur. We reconize a general shape]
)

#v(1em)

=== Change of Basis

To perform PCA, we start with a dataset of femur bone shapes represented as high-dimensional vectors. We note our $N$ femurs $S_i in RR^(3P)$ the shape vector of the $i^(t h)$ femur, where $P$ is the number of points used to represent the shape. The first step is to compute the mean shape vector $macron(S)$:

$ S = 1/N sum_(i=1)^N S_i $

We then center the data by subtracting the mean shape from each femur shape vector and compute the covariance matrix $C$ of the centered data:

$ C = 1/(N-1) sum_(i=1)^N (S_i - macron(S))(S_i - macron(S))^T $

Our goal is to find a change of basis four our centered vector that all our data are uncorrelated. That means we want to find a basis where the covariance matrix is diagonal. This is achieved by finding the eigenvalues and eigenvectors of the covariance matrix $C$. The eigenvectors represent the directions of maximum variance (principal components), and the corresponding eigenvalues indicate the amount of variance captured by each principal component.

#v(1em)

=== Dimensionality Reduction

Once we have the principal components, we can project the original femur shape vectors onto a lower-dimensional subspace spanned by the top $K$ principal components. This is done by selecting the $K$ eigenvectors $v_k$corresponding to the largest eigenvalues $lambda_k$.

Any femur instance $S_i$ in the dataset can be approximated as the mean shape plus a weighted sum of the principal components.

$ S_i approx macron(S) + sum_(k=1)^K w_k v_k $

where $w_k$ correspond to the standard déviations along each principal component direction.

=== Results, visualisation and limitations

Doing the pca on our femur dataset, allow us to reduce the dimensionality from 54873 (3*18291) to 10 while still capturing a significant amount of the variance in the data.

#v(1em)

However, PCA has its limitations. It assumes that the data lies on a linear subspace, which may not always be the case for complex shapes like femur bones. Additionally, PCA is sensitive to outliers and may not capture non-linear relationships in the data. To address these limitations, we also explored non-linear dimensionality reduction techniques, such as Neural Networks.

