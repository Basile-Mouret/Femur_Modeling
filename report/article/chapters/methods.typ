// detailed description of the methods chosen to solve the challenge.

== Modelisation of Neurons

A single artificial neuron receives an input of size $n$, each component associated with a weight. The neuron computes a weighted sum of these inputs and adds a bias term. Mathematically, this can be expressed as:

$ f(x) = b + sum_(k=1)^n w_k a_k $

However, this result is just a linear combination of the inputs. To introduce non-linearity into the model, an activation function is applied to this weighted sum. 

#v(1em)

#figure(
  image("../../fig/sigmoid.png", width: 70%),
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

== Multi-threading

Since the NN is quiet slow to train, we implemented a multi-threading system to speed up the process. \
Firstly we try to create a thread for each operation that can be parallelized. But the overhead created by the creation of threads is too important compared to the time saved (we began with vector addition):

```cpp
template<typename T>
void add_elems(size_t i, const Vector<T>& a, const Vector<T>& b, Vector<T>& result) {
    result.setCoeff(i, a.m_data(i) + b.m_data(i));
}

template<typename T>
Vector<T> Vector<T>::operator+(const Vector<T> &other){
    if (m_size != other.m_size) {
        std::cout << "ERROR: Vectors must be of the same size for addition." << std::endl;
        return *this;
    }
    Vector<T> result(m_size);
    
    std::vector<std::thread> threads;

    for (size_t i = 0; i < m_size; ++i) {
        threads.push_back(std::thread(add_elems<T>, i, std::ref(*this), std::ref(other), std::ref(result)));
    }

    for (auto& t : threads) {
        t.join();
    }

    return result;
}
```

With this approach, the time taken to train our network is too long because of the large number of threads created, we can't train in a reasonable time.

So, we decided to split the operations in a fixed number of threads (depended of the computer threads, basically 4 or 8). Each thread will compute a part of the result vector.


Notes :
Pour testNeuralNetwork:
- sans multithreading : backward 65% of all, forward 16% of all (25.5% de backward)
- avec : backward 75% of all, forward 68% of all, 91% of parent
pas utile --> plus lent

Pour le forward
Multithreading a faire sur la multiplicaton Matrice vecteur, vecteur + vecteur