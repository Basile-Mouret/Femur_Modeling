// detailed description of the methods chosen to solve the challenge.
== Neural Networks

Neural Networks are a class of machine learning models inspired by the structure and function of biological neural networks. They are particularly well-suited for modeling complex, non-linear relationships in data. In this section, we describe the architecture and training process of the neural network implemented for our shape analysis task.

=== Modelisation of Neurons

A single artificial neuron receives an input $x$ of size $n$, $x in bb(R)^n$. Each component associated with a weight. The neuron computes a weighted sum of these inputs and adds a bias term. Mathematically, this can be expressed as:

$ f(x) = b + sum_(k=1)^n w_k x_k quad x = mat(x_1, dots , x_n) $

However, this result is just a linear combination of the inputs. To introduce non-linearity into the model, an activation function is applied to this weighted sum. 

#v(1em)

#figure(
  image("../../fig/sigmoid.png", width: 70%),
  caption: [Sigmoid activation function $Phi(x) = 1 / (1 + e^(-x))$]
)
#v(1em)

*Remark:* The sigmoid function is one of many activation functions used in neural networks. Others include ReLU (Rectified Linear Unit), Tanh, or Softmax, each with its own advantages and applications.

#v(1em)

Finally, the output of the neuron is given by the activation function applied to the weighted sum:

$ (Phi compose f)(x) = Phi(f(x)) $

#v(1em)
#figure(
  [PLACEHOLDER for Neuron Diagram],
  caption: [Structure of an Artificial Neuron]
)


=== Layers and Network Architecture

Artificial neurons are organized into layers to form a neural network. A typical feedforward neural network consists of an input layer, one or more hidden layers, and an output layer. Each layer contains multiple neurons, and the output of one layer serves as the input to the next layer.

#v(1em)
#figure(
  image("../../fig/NN_Diagram.png", width: 70%),
  caption: [Structure of a Neural Network]
)

However, the weights and biases of the neurons are initially set to random values and need to be optimized through a training process.

=== Backpropagation and Training

Training a neural network involves adjusting the weights and biases to minimize the difference between the predicted outputs and the actual target values. This is typically done using a method called backpropagation combined with an optimization algorithm like gradient descent.

The training process consists of the following steps:

1. `Forward Pass`: The input data is passed through the network layer by layer to compute the output predictions.

2. `Loss Calculation`: The loss function quantifies the difference between the predicted outputs and the actual target values. 

3. `Backward Pass`: The gradients of the loss function with respect to each weight and bias are computed using the chain rule of calculus. This process is known as backpropagation.

4. `Weight Update`: The weights and biases are updated using the computed gradients and a learning rate, which determines the step size for each update.

=== Neural Network Implementation

For our problem, our goal is to reduce the dimensionality of the input data (femur shapes represented as high-dimensional vectors) to a lower-dimensional representation while preserving as much information as possible. 

We tried several architectures but each one had this same characteristics:

- Input layer size: 54873 (number of coordinates of the femur mesh)
- Output layer size: 54873 (reconstructed femur mesh)
- Hidden layers: several layers with decreasing and then increasing sizes to create a bottleneck effect, forcing the network to learn a compressed representation of the input data.
- A latent space size of 10.

To train the network, we used a dataset of femur shapes, splitting it into training and validation sets. During the training process, we gave the network femur shapes as input and used the same shapes as target outputs, effectively training the network to reconstruct the input data.

After the training, we could use the encoder part of the network to obtain a low-dimensional representation of any femur shape, and the decoder part to reconstruct the femur shape from its low-dimensional representation.

== Multi-threading

Since the NN is quiet slow to train, we implemented a multi-threading system to speed up the process. \
Firstly we try to create a thread for each operation that can be parallelized. But the overhead created by the creation of threads is too important compared to the time saved (we began with vector addition):

With this approach, the time taken to train our network is too long because of the large number of threads created, we can't train in a reasonable time.

So, we decided to split the operations in a fixed number of threads (depended of the computer threads, basically 4 or 8). Each thread will compute a part of the result vector.


== PCA

