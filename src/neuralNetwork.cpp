#include "neuralNetwork.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>

// Constructor
template <typename T>
NeuralNetwork<T>::NeuralNetwork(const std::vector<size_t>& layers, T learningRate)
    : m_layers(layers), m_learningRate(learningRate), 
      m_activation("sigmoid"), m_loss("meanSquaredError") {
    
    if (layers.size() < 2) {
        std::cerr << "Error: Network must have at least 2 layers (input and output)" << std::endl;
        return;
    }
    
    // Initialize weight matrices and bias vectors
    // For each pair of consecutive layers
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        // Create weight matrix of size (layers[i+1] x layers[i])
        Matrix2D<T> weights(layers[i + 1], layers[i]);
        m_weights.push_back(weights);
        
        // Create bias vector of size layers[i+1]
        Vector<T> bias(layers[i + 1]);
        m_biases.push_back(bias);
    }
    
    // Initialize weights
    initializeWeights();
}

template <typename T>
NeuralNetwork<T>::NeuralNetwork(const std::string& filename) {
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
    }

    // Load architecture
    size_t numLayers;
    file >> numLayers;

    for (size_t i = 0; i < numLayers; ++i) {
        size_t size;
        file >> size;
        m_layers.push_back(size);
    }

    // Load learning rate
    file >> m_learningRate;
    
    // Load activation and loss function names
    file >> m_activation;
    file >> m_loss;

    // Load weights and biases
    for (size_t layer = 0; layer < numLayers - 1; ++layer) {
        // Load dimensions
        size_t rows, cols;
        file >> rows >> cols;

        Matrix2D<T> weights(rows, cols);

        // Load weights
        for (size_t i = 0; i < rows; ++i) {
            for (size_t j = 0; j < cols; ++j) {
                T value;
                file >> value;
                weights.setCoeff(i, j, value);
            }
        }
        m_weights.push_back(weights);

        // Load biases
        Vector<T> bias(rows);
        for (size_t i = 0; i < rows; ++i) {
            T value;
            file >> value;
            bias.setCoeff(i, value);
        }
        m_biases.push_back(bias);
    }

    file.close();

    std::cout << "Network loaded from " << filename << std::endl;
}

// Destructor
template <typename T>
NeuralNetwork<T>::~NeuralNetwork() {
    // Automatic cleanup thanks to std::vector destructors
}

// Weight initialization (Xavier/He initialization)
template <typename T>
void NeuralNetwork<T>::initializeWeights(int seed) {
    std::mt19937 gen(seed);
    
    for (size_t layer = 0; layer < m_weights.size(); ++layer) {
        size_t inputSize = m_layers[layer];
        size_t outputSize = m_layers[layer + 1];
        
        // Xavier initialization: variance = 2 / (inputSize + outputSize)
        T stddev = std::sqrt(2.0 / (inputSize + outputSize));
        std::normal_distribution<T> dist(0.0, stddev);
        
        // Initialize weights
        for (size_t i = 0; i < outputSize; ++i) {
            for (size_t j = 0; j < inputSize; ++j) {
                m_weights[layer].setCoeff(i, j, dist(gen));
            }
        }
        
        // Initialize biases to 0
        for (size_t i = 0; i < outputSize; ++i) {
            m_biases[layer].setCoeff(i, 0.0);
        }
    }
}

// Forward propagation
template <typename T>
Vector<T> NeuralNetwork<T>::forward(const Vector<T>& input) {
    if (input.getSize() != m_layers[0]) {
        std::cerr << "Error: Input size (" << input.getSize() 
                  << ") does not match input layer size (" 
                  << m_layers[0] << ")" << std::endl;
        return Vector<T>(m_layers.back());  // Return zero vector
    }
    
    // Reset activation vectors
    m_activations.clear();
    m_preActivations.clear();
    
    // First activation is the input itself
    m_activations.push_back(input);
    
    Vector<T> currentActivation = input;
    
    // Propagate through all layers
    for (size_t layer = 0; layer < m_weights.size(); ++layer) {
        // Compute z = W * a + b
        Vector<T> z = m_weights[layer] * currentActivation;
        z = z + m_biases[layer];
        
        m_preActivations.push_back(z);
        
        // Apply activation function (sigmoid)
        currentActivation = m_activationFunction.sigmoid(z);
        m_activations.push_back(currentActivation);
    }
    
    return currentActivation;
}

// Backward propagation with weight updates
template <typename T>
T NeuralNetwork<T>::backward(const Vector<T>& input, const Vector<T>& target) {
    // Forward pass
    Vector<T> output = forward(input);
    
    // Compute loss (meanSquaredError)
    T loss = m_lossFunction.meanSquaredError(output, target);
    
    // Compute loss gradient with respect to output
    Vector<T> dLoss = m_lossFunction.meanSquaredErrorDerivative(output, target);
    
    // Backpropagation
    std::vector<Vector<T>> deltas;
    
    // Output layer
    size_t lastLayer = m_weights.size() - 1;
    Vector<T> sigmoidDeriv = m_activationFunction.sigmoidDerivative(m_preActivations[lastLayer]);
    
    // Delta of the last layer: dLoss * sigmoid'(z) - using Hadamard product
    Vector<T> delta = dLoss.hadamard(sigmoidDeriv);
    deltas.push_back(delta);
    
    // Backpropagation to hidden layers
    for (int layer = lastLayer - 1; layer >= 0; --layer) {
        // delta[layer] = (W[layer+1]^T * delta[layer+1]) * sigmoid'(z[layer])
        
        // Compute W^T * delta using transpose
        Matrix2D<T> W_transpose = m_weights[layer + 1].transpose();
        Vector<T> weightedDelta = W_transpose * deltas[lastLayer - layer - 1];
        
        // Multiply by sigmoid'(z) using Hadamard product
        sigmoidDeriv = m_activationFunction.sigmoidDerivative(m_preActivations[layer]);
        Vector<T> currentDelta = weightedDelta.hadamard(sigmoidDeriv);
        deltas.push_back(currentDelta);
    }
    
    // Reverse deltas order (computed from end to beginning)
    std::reverse(deltas.begin(), deltas.end());
    
    // Update weights and biases
    for (size_t layer = 0; layer < m_weights.size(); ++layer) {
        // Update weights: W -= learningRate * (delta * a^T)
        // delta * a^T is the outer product of delta and activation
        Matrix2D<T> gradient = deltas[layer].outerProduct(m_activations[layer]);
        Matrix2D<T> weightUpdate = gradient * m_learningRate;
        m_weights[layer] = m_weights[layer] - weightUpdate;
        
        // Update biases: b -= learningRate * delta
        m_biases[layer] = m_biases[layer] - (deltas[layer] * m_learningRate);
    }
    
    return loss;
}

// Training
template <typename T>
std::vector<T> NeuralNetwork<T>::train(const std::vector<Vector<T>>& inputs, 
                                       const std::vector<Vector<T>>& targets, 
                                       size_t epochs, 
                                       bool verbose) {
    if (inputs.size() != targets.size()) {
        std::cerr << "Error: Number of inputs and targets do not match" << std::endl;
        return std::vector<T>();
    }
    
    std::vector<T> lossHistory;
    
    for (size_t epoch = 0; epoch < epochs; ++epoch) {
        T totalLoss = 0;
        
        // Train on each example
        for (size_t i = 0; i < inputs.size(); ++i) {
            T loss = backward(inputs[i], targets[i]);
            totalLoss += loss;
        }
        
        T avgLoss = totalLoss / inputs.size();
        lossHistory.push_back(avgLoss);
        
        if (verbose && (epoch % 100 == 0 || epoch == epochs - 1)) {
            std::cout << "Epoch " << std::setw(5) << epoch 
                      << " - Loss: " << std::fixed << std::setprecision(6) << avgLoss 
                      << std::endl;
        }
    }
    
    return lossHistory;
}

// Prediction
template <typename T>
Vector<T> NeuralNetwork<T>::predict(const Vector<T>& input) {
    return forward(input);
}

// Save network to file
template <typename T>
bool NeuralNetwork<T>::save(const std::string& filename) const {
    std::ofstream file(filename);
    
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return false;
    }
    
    // Save architecture
    file << m_layers.size() << std::endl;
    for (size_t size : m_layers) {
        file << size << " ";
    }
    file << std::endl;
    
    // Save learning rate
    file << m_learningRate << std::endl;
    
    // Save activation and loss function names
    file << m_activation << std::endl;
    file << m_loss << std::endl;
    
    // Save weights and biases
    for (size_t layer = 0; layer < m_weights.size(); ++layer) {
        // Save weight matrix dimensions
        file << m_weights[layer].getSizeRows() << " " 
             << m_weights[layer].getSizeCols() << std::endl;
        
        // Save weights
        for (size_t i = 0; i < m_weights[layer].getSizeRows(); ++i) {
            for (size_t j = 0; j < m_weights[layer].getSizeCols(); ++j) {
                file << m_weights[layer](i, j) << " ";
            }
            file << std::endl;
        }
        
        // Save biases
        for (size_t i = 0; i < m_biases[layer].getSize(); ++i) {
            file << m_biases[layer](i) << " ";
        }
        file << std::endl;
    }
    
    file.close();
    
    std::cout << "Network saved to " << filename << std::endl;
    
    return true;
}

// Accessors
template <typename T>
const std::vector<size_t>& NeuralNetwork<T>::getLayers() const {
    return m_layers;
}

template <typename T>
T NeuralNetwork<T>::getLearningRate() const {
    return m_learningRate;
}

template <typename T>
void NeuralNetwork<T>::setLearningRate(T learningRate) {
    m_learningRate = learningRate;
}

template <typename T>
const std::vector<Matrix2D<T>>& NeuralNetwork<T>::getWeights() const {
    return m_weights;
}

template <typename T>
const std::vector<Vector<T>>& NeuralNetwork<T>::getBiases() const {
    return m_biases;
}

template <typename T>
const std::string& NeuralNetwork<T>::getActivation() const {
    return m_activation;
}

template <typename T>
const std::string& NeuralNetwork<T>::getLoss() const {
    return m_loss;
}

// Explicit template instantiation for common types
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;
