== Neural Networks

Neural Networks are a class of machine learning models inspired by the structure and function of biological neural networks. They are particularly well-suited for modeling complex, non-linear relationships in data. In this section, we describe the architecture and training process of the neural network implemented for our shape analysis task.

=== Modelisation of Neurons

A single artificial neuron receives an input $x$ of size $n$, $x in bb(R)^n$. Each component associated with a weight. The neuron computes a weighted sum of these inputs and adds a bias term. Mathematically, this can be expressed as:

$ f(x) = b + sum_(k=1)^n w_k x_k quad x = mat(x_1, dots , x_n) $

However, this result is just a linear combination of the inputs. To introduce non-linearity into the model, an activation function is applied to this weighted sum. 

#v(1em)

#figure(
  image("/resources/img/sigmoid.png", width: 70%),
  caption: [Sigmoid activation function $Phi(x) = 1 / (1 + e^(-x))$]
)
#v(1em)

*Remark:* The sigmoid function is one of many activation functions used in neural networks. Others include ReLU (Rectified Linear Unit), Tanh, or Softmax, each with its own advantages and applications.

#v(1em)

Finally, the output of the neuron is given by the activation function applied to the weighted sum:

$ (Phi compose f)(x) = Phi(f(x)) $

#v(1em)
#grid(
  columns: (1fr, 1fr),
  gutter: 1em,
  [
  #figure(
  image("/resources/img/Neuron_diagram.png", width: 71%),
  caption: [Structure of an Artificial Neuron]
  )],
  [#figure(
  image("/resources/img/NN_Diagram.png", width: 100%),
  caption: [Structure of a Neural Network])]
)

#v(1em)
=== Layers and Network Architecture

Artificial neurons are organized into layers to form a neural network. A typical feedforward neural network consists of an input layer, one or more hidden layers, and an output layer. Each layer contains multiple neurons, and the output of one layer serves as the input to the next layer. However, the weights and biases of the neurons are initially set to random values and need to be optimized through a training process.

=== Backpropagation and Training

Training a neural network involves adjusting the weights and biases to minimize the difference between the predicted outputs and the actual target values. This is typically done using a method called backpropagation combined with an optimization algorithm like gradient descent.

The training process consists of the following steps:

1. `Forward Pass`: The input data is passed through the network layer by layer to compute the output predictions.

2. `Loss Calculation`: The loss function quantifies the difference between the predicted outputs and the actual target values. 

3. `Backward Pass`: The gradients of the loss function with respect to each weight and bias are computed using the chain rule of calculus. This process is known as backpropagation.

4. `Weight Update`: The weights and biases are updated using the computed gradients and a learning rate, which determines the step size for each update.

=== Neural Network Implementation

For this project, we implemented a Neural Network from scratch in `C++`. This gave us a deep understanding of the method and the underlying computations. We started by creating a `Vector` and `Matrix2D` class for which we implemented all the necessary linear algebra methods for a neural network. We used the `Eigen` library to store the data as we were only interested in the mathematical implementation. We then created a `Neural Network` class defining fully connected neural networks with specific activation and loss functions. This class also has the methods to do a forward pass of the neural network and train it using backpropagation.

=== Autoencoder Structure

In order to reduce the dimensionality and to discover underlying relationships in the data we used an Autoencoder structure.
This is a type of fully connected neural network which tries to regenerate the input data while passing through a very small layer.
This layer called the latent space acts as a bottleneck for information transfer between the encoder (the part before the latent space) and the decoder (the part after it).
Our input and output layers consist of 54873 neurons as we had 18291 three dimensional points for each femur.
By empirical testing, we chose a latent space of size 10 as it seemed enough to capture the main component of the dataset.
As we used our own implementation of neural networks, which isn't as efficient a state of the art libraries, we had to aggressively compress each layer, going from the full 54873 down to 256, 32 and finally 10 neurons for the latent space.
Furthermore, in order to get faster training, we did some preprocessing by subtracting the mean femur before feeding them to the autoencoder.

#figure(
  box(width: 100%)[
    #grid(
      columns: (auto, auto, 19em, auto, auto),
      align: (center + horizon),
      column-gutter: 1em, // Added for better spacing between elements

      // 1. Original
      stack(dir: ttb, spacing: 0.5em,
        image("../resources/img/original_L_Femur_11.png"),
        [Original\ Femur]
      ),

      // 2. Preprocessing
      stack(dir: ttb, spacing: 0.5em,
        $arrow.long$, 
        text(size: 0.8em)[Preprocessing]
      ),

      // 3. Network (Big)
      image("../resources/img/autoencoder.svg", width: 19em),

      // 4. Postprocessing
      stack(dir: ttb, spacing: 0.5em,
        $arrow.long$,
        text(size: 0.8em)[Postprocessing]
      ),

      // 5. Reconstructed
      stack(dir: ttb, spacing: 0.5em,
        image("../resources/img/reconstructed_L_Femur_11.png"),
        [Reconstructed\ Femur]
      )
    )
  ],
  caption: [Autoencoder pipeline for femur mesh reconstruction],
  supplement: [Figure],
) <autoencoder-fig>

We used the Mean Squared error for the loss function as all the point were corresponding to one another. We also tried different activation functions, and got the best results with LeakyReLU and tanh.

As we implemented the neural network and linear algebra ourselves, the training took a long time. We then focused on increasing performance.

== Multi-threading

Since training the neural network is quite slow, we used multi-threading to speed up the process.
We focused on parallelizing the matrix-vector multiplication (MatVec) operations, as these are computationally intensive and benefit the most from parallel execution.

=== With `std::threads`
Our first approach was to use `std::threads` from the C++ standard library.

==== Naive method
At first, we naively tried to create a thread for each operation that could be parallelized. However, the overhead from creating so many threads was far greater than any time saved. For example, for the largest matrix in our network ($54873 times 512 = 28,094,976$ parameters), we would have created 28,094,976 threads, which is obviously not feasible. With this approach, our neural network was actually slower than the single-threaded version.

==== Improved approach <better-approach>
We then decided to split the computation among a fixed number of threads (depending on the number of CPU cores, typically 8 or 16). Each thread computes a portion of the result vector: the first thread computes the first $"result.size"() / "num_threads"$ elements, the second thread the next $"result.size()" / "num_threads"$ elements, and so on.

This approach worked much better, and we observed a significant speedup in the training time of our neural network.
However, there was still some overhead due to thread creation and destruction at each MatVec operation.

==== Thread pool <thread-pool>
As noted in @better-approach, we were still creating and destroying too many threads. The best solution would be to implement a thread pool, where a fixed number of threads are created at the start of the program and reused for multiple MatVec operations.
Unfortunately, due to time constraints, we did not implement this method, which could have further improved the performance of our neural network training.

=== OpenMP
A second approach was to use OpenMP, a popular API for parallel programming (see @openmp). It provides a simple and efficient way to parallelize code, as it is close to sequential code and automatically manages thread creation and workload distribution.

=== Performance Comparison

#figure(
  image("/resources/img/perf_multithreading.png", width: 100%),
  caption: [Performance comparison between single-threaded and multi-threaded implementations.
   *Note:* The number at the end of epoch_times corresponds to the threshold parameter, which is the number of parameters in the weight matrix above which we use multithreading.]
) <perf-multithreading>

As shown in @perf-multithreading, using multi-threading significantly reduces the time required to train the neural network: we gain about 3 seconds per epoch, which is quite significant since we train our neural network for hundreds of epochs. \
Even though we achieved better results with this multi-threaded approach, as mentioned in @thread-pool, we could have obtained even greater improvements by implementing a thread pool with `std::threads`.

Overall, the multi-threaded implementation shows a clear advantage over the single-threaded version, especially as the size of the dataset increases. This demonstrates the effectiveness of parallelizing computationally intensive operations in our neural network training process.

*Remark:* We observed that the time taken by our neural network to train with OpenMP multi-threading or with our `std::threads` implementation is quite similar. The OpenMP method likely uses an algorithm similar to the one we implemented with `std::threads`.

*Remark 2:* As expected, with the threshold parameter set to 100,000,000, all computations are single-threaded (since the largest matrix in the neural network contains 28,094,976 parameters), so the training time matches that of the single-threaded implementation.

#include "memory_allocations.typ"

#include "results_visu.typ"
