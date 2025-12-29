#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <vector>
#include <string>
#include <fstream>
#include "linalg.hpp"
#include "neuralNetworkFunctions.hpp"

template <typename T>
class NeuralNetwork {
private:
    // Network architecture
    std::vector<size_t> m_layers;
    
    // Weights and biases for each layer
    std::vector<Matrix2D<T>> m_weights;
    std::vector<Vector<T>> m_biases;
    
    // Function objects for activation and loss computation
    ActivationFunction<T> m_activationFunction;
    LossFunction<T> m_lossFunction;
    
    // Function names as strings for identification
    std::string m_activation;  // TODO : support multiple activation functions
    std::string m_loss;        // TODO : support multiple loss functions
    
    // Learning rate
    T m_learningRate;
    
    // For forward propagation (stores activations and pre-activation values)
    std::vector<Vector<T>> m_activations;
    std::vector<Vector<T>> m_preActivations;

public:
    // Constructors
    
    /**
     * @brief Constructs a neural network with the specified architecture
     * @param layers Vector containing the size of each layer [input, hidden..., output]
     * @param learningRate Learning rate for gradient descent (default: 0.01)
     */
    NeuralNetwork(const std::vector<size_t>& layers, T learningRate = 0.01);
    
    /**
     * @brief Destructor
     */
    ~NeuralNetwork();
    
    // Main methods
    
    /**
     * @brief Initializes weights and biases randomly using Xavier/He initialization
     * @param seed Random seed for reproducibility (default: 100)
     */
    void initializeWeights(int seed = 100);
    
    /**
     * @brief Forward propagation: computes the network output for a given input
     * @param input Input vector
     * @return Output vector from the network
     */
    Vector<T> forward(const Vector<T>& input);
    
    /**
     * @brief Backward propagation: computes gradients and updates weights
     * @param input Input vector
     * @param target Expected output vector
     * @return Loss value
     */
    T backward(const Vector<T>& input, const Vector<T>& target);
    
    /**
     * @brief Trains the network on a dataset
     * @param inputs Training input data
     * @param targets Training target outputs
     * @param epochs Number of training epochs
     * @param verbose Whether to display training information (default: true)
     * @return Vector containing loss history
     */
    std::vector<T> train(const std::vector<Vector<T>>& inputs, 
                         const std::vector<Vector<T>>& targets, 
                         size_t epochs, 
                         bool verbose = true);
    
    /**
     * @brief Predicts the output for a given input
     * @param input Input vector
     * @return Network prediction
     */
    Vector<T> predict(const Vector<T>& input);
    
    // Save and load
    
    /**
     * @brief Saves the network to a file
     * @param filename Path to the output file
     * @return true if save was successful
     */
    bool save(const std::string& filename) const;
    
    /**
     * @brief Loads a network from a file
     * @param filename Path to the input file
     * @return true if load was successful
     */
    bool load(const std::string& filename);
    
    // Accessors
    
    /**
     * @brief Gets the network architecture
     * @return Vector containing layer sizes
     */
    const std::vector<size_t>& getLayers() const;
    
    /**
     * @brief Gets the learning rate
     * @return Learning rate value
     */
    T getLearningRate() const;
    
    /**
     * @brief Sets the learning rate
     * @param learningRate New learning rate value
     */
    void setLearningRate(T learningRate);
    
    /**
     * @brief Gets the weight matrices
     * @return Vector of weight matrices
     */
    const std::vector<Matrix2D<T>>& getWeights() const;
    
    /**
     * @brief Gets the bias vectors
     * @return Vector of bias vectors
     */
    const std::vector<Vector<T>>& getBiases() const;
    
    /**
     * @brief Gets the activation function name
     * @return Name of the activation function (e.g., "sigmoid")
     */
    const std::string& getActivation() const;
    
    /**
     * @brief Gets the loss function name
     * @return Name of the loss function (e.g., "meanSquaredError")
     */
    const std::string& getLoss() const;
};

#endif // NEURAL_NETWORK_HPP
