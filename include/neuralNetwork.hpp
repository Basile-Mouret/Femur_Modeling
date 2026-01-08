/**
 * @file neuralNetwork.hpp
 * @brief Neural network implementation for feedforward multilayer perceptron
 * @details This file contains a template class for creating, training, and using
 *          feedforward neural networks with customizable architectures, activation
 *          functions, and loss functions. Supports forward/backward propagation,
 *          training, prediction, and model persistence.
 */

#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>
#include <fstream>
#include "linalg.hpp"
#include "neuralNetworkFunctions.hpp"

/**
 * @class NeuralNetwork
 * @brief Template class implementing a feedforward neural network
 * @tparam T Data type for computations (typically float or double)
 * 
 * This class provides a complete implementation of a multilayer perceptron (MLP)
 * neural network with configurable architecture. It supports training via
 * backpropagation and gradient descent, as well as inference, model saving,
 * and loading.
 * 
 * @note Currently supports sigmoid activation and MSE loss (TODO: multiple functions)
 */
template <typename T>
class NeuralNetwork {
private:
    // Network architecture
    std::vector<size_t> m_layers;                   ///< Layer sizes [input, hidden..., output]
    
    // Weights and biases for each layer
    std::vector<Matrix2D<T>> m_weights;             ///< Weight matrices for each connection
    std::vector<Vector<T>> m_biases;                ///< Bias vectors for each layer
    
    // Function objects for activation and loss computation
    ActivationFunction<T> m_activationFunction;     ///< Activation function object
    LossFunction<T> m_lossFunction;                 ///< Loss function object
    
    // Function names as strings for identification
    std::string m_activation;                       ///< Activation function name (TODO: multiple)
    std::string m_loss;                             ///< Loss function name (TODO: multiple)
    
    // Learning rate
    T m_learningRate;                               ///< Learning rate for gradient descent
    
    // For forward propagation (stores activations and pre-activation values)
    std::vector<Vector<T>> m_activations;           ///< Layer activations during forward pass
    std::vector<Vector<T>> m_preActivations;        ///< Pre-activation values (z = Wx + b)

public:
    // Constructors
    
    /**
     * @brief Constructs a neural network with the specified architecture
     * 
     * Creates a feedforward neural network with the given layer sizes.
     * The first element is the input layer size, intermediate elements are
     * hidden layer sizes, and the last element is the output layer size.
     * 
     * Example: {784, 128, 64, 10} creates a network with:
     * - Input: 784 neurons
     * - Hidden layer 1: 128 neurons
     * - Hidden layer 2: 64 neurons
     * - Output: 10 neurons
     * 
     * @param layers Vector containing the size of each layer [input, hidden..., output]
     * @param learningRate Learning rate for gradient descent (default: 0.01)
     */
    NeuralNetwork(const std::vector<size_t>& layers, T learningRate = 0.01);

    /**
     * @brief Constructs a neural network from a saved file
     * 
     * Creates a neural network by loading a previously saved model from a file.
     * This constructor restores the network architecture, weights, biases, and
     * hyperparameters from the specified file.
     * 
     * @param filename Path to the file containing the saved network model
     */
    NeuralNetwork(const std::string& filename);

    /**
     * @brief Destructor
     * 
     * Cleans up allocated resources.
     */
    ~NeuralNetwork();
    
    // Main methods
    
    /**
     * @brief Initializes weights and biases with random values
     * 
     * Uses Xavier/He initialization for optimal weight initialization.
     * Weights are initialized with small random values scaled by the
     * layer dimensions to prevent vanishing/exploding gradients.
     * 
     * @param seed Random seed for reproducibility (default: 100)
     */
    void initializeWeights(int seed = 100);
    
    /**
     * @brief Forward propagation through the network
     * 
     * Computes the network output for a given input by propagating
     * through all layers. Stores intermediate activations and
     * pre-activation values for use in backpropagation.
     * 
     * @param input Input feature vector
     * @return Output vector from the network's final layer
     */
    Vector<T> forward(const Vector<T>& input);
    
    /**
     * @brief Backward propagation and weight update
     * 
     * Computes gradients via backpropagation and updates weights and
     * biases using gradient descent. Must be called after forward().
     * 
     * @param input Input feature vector (same as used in forward())
     * @param target Expected output (ground truth) vector
     * @return Loss value for this training example
     */
    T backward(const Vector<T>& input, const Vector<T>& target);
    
    /**
     * @brief Trains the network on a dataset
     * 
     * Performs training for a specified number of epochs using the
     * provided input-target pairs. Each epoch processes all training
     * examples once with forward and backward propagation.
     * 
     * @param inputs Training input feature vectors
     * @param targets Training target output vectors (ground truth)
     * @param epochs Number of complete passes through the dataset
     * @param verbose Whether to display training progress and loss (default: true)
     * @return Vector containing the loss value at each epoch
     */
    std::vector<T> train(const std::vector<Vector<T>>& inputs, 
                         const std::vector<Vector<T>>& targets, 
                         size_t epochs, 
                         bool verbose = true);
    
    /**
     * @brief Predicts the output for a given input
     * 
     * Performs inference by running forward propagation without
     * storing intermediate values for backpropagation.
     * 
     * @param input Input feature vector
     * @return Network prediction output vector
     */
    Vector<T> predict(const Vector<T>& input);
    
    // Save and load
    
    /**
     * @brief Saves the network architecture and weights to a file
     * 
     * Serializes the complete network state including layer sizes,
     * weights, biases, and hyperparameters to a file for later loading.
     * 
     * @param filename Path to the output file
     * @return true if save was successful, false otherwise
     */
    bool save(const std::string& filename) const;

    // Accessors
    
    /**
     * @brief Gets the network architecture layer sizes
     * 
     * Returns a reference to the vector containing the size of each layer.
     * 
     * @return Const reference to vector of layer sizes
     */
    const std::vector<size_t>& getLayers() const;
    
    /**
     * @brief Gets the current learning rate
     * 
     * Returns the learning rate used for gradient descent optimization.
     * 
     * @return Learning rate value
     */
    T getLearningRate() const;
    
    /**
     * @brief Sets a new learning rate
     * 
     * Updates the learning rate for gradient descent. Can be used to
     * implement learning rate schedules or adaptive learning rates.
     * 
     * @param learningRate New learning rate value (should be positive)
     */
    void setLearningRate(T learningRate);
    
    /**
     * @brief Gets the weight matrices
     * 
     * Returns all weight matrices connecting the layers.
     * Matrix i connects layer i to layer i+1.
     * 
     * @return Const reference to vector of weight matrices
     */
    const std::vector<Matrix2D<T>>& getWeights() const;
    
    /**
     * @brief Gets the bias vectors
     * 
     * Returns all bias vectors for the layers.
     * Vector i contains biases for layer i+1.
     * 
     * @return Const reference to vector of bias vectors
     */
    const std::vector<Vector<T>>& getBiases() const;
    
    /**
     * @brief Gets the activation function name
     * 
     * Returns the name identifier of the activation function currently in use.
     * 
     * @return Name of the activation function (e.g., "sigmoid")
     */
    const std::string& getActivation() const;
    
    /**
     * @brief Gets the loss function name
     * 
     * Returns the name identifier of the loss function currently in use.
     * 
     * @return Name of the loss function (e.g., "meanSquaredError")
     */
    const std::string& getLoss() const;
};

#endif // NEURAL_NETWORK_HPP
