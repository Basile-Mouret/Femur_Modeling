= Neural Network Structure
#v(1em)

The human brain is composed over 80 billion neurons, interconnected through trillions of synapses. This intricate network allows for complex processing and communication, enabling functions such as perception, cognition, and motor control. Artificial neural networks (ANNs) are computational models inspired by the structure and function of biological neural networks. They consist of layers of interconnected nodes, or "neurons," that process information in a manner similar to the human brain.

== Modelisation of Neurons

A single artificial neuron receives an input of size $n$, each component associated with a weight. The neuron computes a weighted sum of these inputs and adds a bias term. Mathematically, this can be expressed as:

$ f(x) = b + sum_(k=1)^n w_k a_k $

However, this result is just a linear combination of the inputs. To introduce non-linearity into the model, an activation function is applied to this weighted sum. 

#v(1em)

#figure(
  image("fig/sigmoid.png", width: 70%),
  caption: [Sigmoid activation function $Phi(x) = 1 / (1 + e^(-x))$]
)

*Remark:* The sigmoid function is one of many activation functions used in neural networks. Others include ReLU (Rectified Linear Unit), Tanh, or Softmax, each with its own advantages and applications.

#v(1em)

Finally, the output of the neuron is given by the activation function applied to the weighted sum:

$ (Phi compose f)(x) = Phi(f(x)) $

#v(1em)
#figure(
  [PLACEHOLDER for Neuron Diagram],
  caption: [Structure of an Artificial Neuron]
)

== Layers and Network Architecture

Artificial neurons are organized into layers to form a neural network. A typical feedforward neural network consists of an input layer, one or more hidden layers, and an output layer. Each layer contains multiple neurons, and the output of one layer serves as the input to the next layer.

#v(1em)
#figure(
  [PLACEHOLDER for Neural Network Diagram],
  caption: [Structure of a Neural Network]
)

However, the weights and biases of the neurons are initially set to random values and need to be optimized through a training process.

== Backpropagation and Training

