#include "neuralNetwork.hpp"
#include <iostream>
#include <random>
#include <cmath>
#include <iomanip>

// Constructor
template <typename T>
NeuralNetwork<T>::NeuralNetwork(const std::vector<size_t>& layers, const std::string& activation, const std::string& loss, const T learningRate)
    : m_layers(layers), m_learningRate(learningRate), 
      m_activation(activation), m_loss(loss) {
    
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
    // Pre allocation of the vectors
    m_activations.resize(m_layers.size(), Vector<T>(0));
    m_deltas.resize(m_layers.size(), Vector<T>(0));
    m_preActivations.resize(m_layers.size() - 1, Vector<T>(0));

    for (size_t i = 0; i < m_layers.size(); ++i) {
        m_activations[i] = Vector<T>(m_layers[i]);
        m_deltas[i] = Vector<T>(m_layers[i]);
        
        // Pre-activations correspond to the output of weights (layers 1 to N)
        if (i < m_layers.size() - 1) {
            m_preActivations[i] = Vector<T>(m_layers[i+1]);
        }
    } 
    // Initialize weights
    initializeWeights();
}

template <typename T>
NeuralNetwork<T>::NeuralNetwork(const std::string& filename) {
    // Check for binary magic number first
    std::ifstream binFile(filename, std::ios::binary);
    if (binFile.is_open()) {
        uint32_t magic;
        if (binFile.read(reinterpret_cast<char*>(&magic), sizeof(magic))) {
            if (magic == 0x4E4E4249) {
                binFile.close();
                loadBinary(filename);
                return;
            }
        }
        binFile.close();
    }

    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    // Load architecture
    size_t numLayers;
    if (!(file >> numLayers)) return;
    if (!(file >> numLayers)) return;

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
                weights(i, j) = value;
            }
        }
        m_weights.push_back(weights);

        // Load biases
        Vector<T> bias(rows);
        for (size_t i = 0; i < rows; ++i) {
            T value;
            file >> value;
            bias(i) = value;
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
                m_weights[layer](i, j) = dist(gen);
            }
        }
        
        // Initialize biases to 0
        for (size_t i = 0; i < outputSize; ++i) {
            m_biases[layer](i) = 0.0;
        }
    }
}

// Forward propagation
template <typename T>
Vector<T> NeuralNetwork<T>::forward(const Vector<T>& input, size_t layerIndex) {
    if (input.getSize() != m_layers[layerIndex]) {
        std::cerr << "Error: Input size (" << input.getSize() 
                  << ") does not match layer " << layerIndex << " size (" 
                  << m_layers[layerIndex] << ")" << std::endl;
        return Vector<T>(m_layers.back());  // Return zero vector
    }
    
    
    m_activations[0] = input;
    
    Vector<T> currentActivation = input;
    
    // Propagate through all layers starting from layerIndex
    for (size_t layer = layerIndex; layer < m_weights.size(); ++layer) {
        // Compute z = W * a + b
        Vector<T> z = m_weights[layer] * currentActivation + m_biases[layer];
        
        m_preActivations[layer] = z;
        
        // Apply activation function (sigmoid)
        if(m_activation == "sigmoid")
            currentActivation = m_activationFunction.sigmoid(z);
        else if(m_activation == "tanh")
            currentActivation = m_activationFunction.tanh(z);
        else if(m_activation == "ReLU")
            currentActivation = m_activationFunction.ReLU(z);
        else if(m_activation == "LeakyReLU")
            currentActivation = m_activationFunction.LeakyReLU(z);
        else {
            std::cerr << "Error: Unknown activation function " << m_activation << std::endl;
            return Vector<T>(m_layers.back());
        }
        m_activations[layer+1] = currentActivation;
    }
    
    return currentActivation;
}

// Backward propagation with weight updates
template <typename T>
T NeuralNetwork<T>::backward(const Vector<T>& input, const Vector<T>& target) {
    // Forward pass
    Vector<T> output = forward(input);
    
    // Compute loss (meanSquaredError)
    T loss;
    if(m_loss == "meanSquaredError")
        loss = m_lossFunction.meanSquaredError(output, target);
    else {
        std::cerr << "Error: Unknown loss function " << m_loss << std::endl;
        return T(0);
    }
    
    
    // Compute loss gradient with respect to output
    Vector<T> dLoss(m_layers.back());
    if(m_loss == "meanSquaredError") {
        dLoss = m_lossFunction.meanSquaredErrorDerivative(output, target);
    }
    else {
        std::cerr << "Error: Unknown loss function " << m_loss << std::endl;
        return T(0);
    }
    // Backpropagation
    
    // Output layer
    size_t lastLayerIdx = m_layers.size() - 1;

    Vector<T> deriv(m_layers.back());
    if(m_activation == "sigmoid") {
        deriv = m_activationFunction.sigmoidDerivative(m_preActivations[lastLayerIdx]);
    }
    else if(m_activation == "tanh") {
        deriv = m_activationFunction.tanhDerivative(m_preActivations[lastLayerIdx]);
    }
    else if(m_activation == "ReLU") {
        deriv = m_activationFunction.ReLUDerivative(m_preActivations[lastLayerIdx]);
    }
    else if(m_activation == "LeakyReLU") {
        deriv = m_activationFunction.LeakyReLUDerivative(m_preActivations[lastLayerIdx]);
    }
    else {
        std::cerr << "Error: Unknown activation function " << m_activation << std::endl;
        return loss;
    }
    
    // Delta of the last layer: dLoss * activation'(z) - using Hadamard product
    m_deltas[lastLayerIdx] = dLoss.hadamard(deriv);
    
    // Backpropagation to hidden layers
    for (int i = lastLayerIdx - 1; i > 0; --i) {
        // delta[layer] = (W[layer+1]^T * delta[layer+1]) * sigmoid'(z[layer])
        
        // Compute W^T * delta using mutltiplyTranspose
        Vector<T> weightedDelta = m_weights[i].multiplyTranspose(m_deltas[i+1]);
        
        // Multiply by activation'(z) using Hadamard product
        Vector<T> deriv(m_layers[i]);
        if(m_activation == "sigmoid")
            deriv = m_activationFunction.sigmoidDerivative(m_preActivations[i-1]);
        else if(m_activation == "tanh")
            deriv = m_activationFunction.tanhDerivative(m_preActivations[i-1]);
        else if(m_activation == "ReLU")
            deriv = m_activationFunction.ReLUDerivative(m_preActivations[i-1]);
        else if(m_activation == "LeakyReLU")
            deriv = m_activationFunction.LeakyReLUDerivative(m_preActivations[i-1]);
        else {
            std::cerr << "Error: Unknown activation function " << m_activation << std::endl;
        }

        Vector<T> currentDelta = weightedDelta.hadamard(deriv);
        m_deltas[i] = currentDelta;
    }
    
    // Update weights and biases
    for (size_t layer = 0; layer < m_weights.size(); ++layer) {
        // Update weights: W -= learningRate * (delta * a^T)
        // delta * a^T is the outer product of delta and activation
        m_weights[layer].rank1Update(m_deltas[layer + 1], m_activations[layer], m_learningRate);
        
        m_biases[layer] -= (m_deltas[layer + 1] * m_learningRate);
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
        
        if (verbose && (epoch % 10 == 0 || epoch == epochs - 1)) {
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

template <typename T>
Vector<T> NeuralNetwork<T>::decodeLatent(const Vector<T>& latentVector, size_t layerIndex) {
    if (layerIndex >= m_weights.size()) {
        std::cerr << "Error: layerIndex out of bounds in decodeLatent()" << std::endl;
        return Vector<T>(0);
    }
    
    return forward(latentVector, layerIndex);
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

template <typename T>
bool NeuralNetwork<T>::saveBinary(const std::string& filename) const {
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file for binary saving " << filename << std::endl;
        return false;
    }

    uint32_t magic = 0x4E4E4249; // NNBI
    file.write(reinterpret_cast<const char*>(&magic), sizeof(magic));

    size_t numLayers = m_layers.size();
    file.write(reinterpret_cast<const char*>(&numLayers), sizeof(size_t));
    if (numLayers > 0) {
        file.write(reinterpret_cast<const char*>(m_layers.data()), numLayers * sizeof(size_t));
    }

    file.write(reinterpret_cast<const char*>(&m_learningRate), sizeof(T));

    size_t actLen = m_activation.size();
    file.write(reinterpret_cast<const char*>(&actLen), sizeof(size_t));
    if (actLen > 0)
        file.write(m_activation.c_str(), actLen);

    size_t lossLen = m_loss.size();
    file.write(reinterpret_cast<const char*>(&lossLen), sizeof(size_t));
    if (lossLen > 0)
        file.write(m_loss.c_str(), lossLen);

    for (size_t i = 0; i < m_weights.size(); ++i) {
        size_t rows = m_weights[i].getSizeRows();
        size_t cols = m_weights[i].getSizeCols();
        file.write(reinterpret_cast<const char*>(m_weights[i].getData()), rows * cols * sizeof(T));
        
        file.write(reinterpret_cast<const char*>(m_biases[i].getData()), rows * sizeof(T));
    }

    file.close();
    std::cout << "Network saved to (binary) " << filename << std::endl;
    return true;
}

template <typename T>
void NeuralNetwork<T>::loadBinary(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Cannot open file " << filename << std::endl;
        return;
    }

    uint32_t magic;
    file.read(reinterpret_cast<char*>(&magic), sizeof(magic));
    if (magic != 0x4E4E4249) {
        std::cerr << "Error: Invalid binary format (magic number mismatch)" << std::endl;
        return;
    }

    size_t numLayers;
    file.read(reinterpret_cast<char*>(&numLayers), sizeof(size_t));
    
    m_layers.resize(numLayers);
    if (numLayers > 0) {
        file.read(reinterpret_cast<char*>(m_layers.data()), numLayers * sizeof(size_t));
    }

    file.read(reinterpret_cast<char*>(&m_learningRate), sizeof(T));

    size_t actLen;
    file.read(reinterpret_cast<char*>(&actLen), sizeof(size_t));
    m_activation.resize(actLen);
    if (actLen > 0)
        file.read((char*)&m_activation[0], actLen);

    size_t lossLen;
    file.read(reinterpret_cast<char*>(&lossLen), sizeof(size_t));
    m_loss.resize(lossLen);
    if (lossLen > 0)
        file.read((char*)&m_loss[0], lossLen);

    // clear existing
    m_weights.clear();
    m_biases.clear();

    for (size_t i = 0; i < numLayers - 1; ++i) {
        size_t rows = m_layers[i + 1];
        size_t cols = m_layers[i];
   
        Matrix2D<T> weights(rows, cols);
        file.read(reinterpret_cast<char*>(weights.getData()), rows * cols * sizeof(T));
        m_weights.push_back(weights);

        Vector<T> bias(rows);
        file.read(reinterpret_cast<char*>(bias.getData()), rows * sizeof(T));
        m_biases.push_back(bias);
    }
    
    file.close();
    std::cout << "Network loaded from (binary) " << filename << std::endl;
}

// ============================================================================
// LinearOutputNeuralNetwork Implementation
// ============================================================================

// Constructor
template <typename T>
LinearOutputNeuralNetwork<T>::LinearOutputNeuralNetwork(const std::vector<size_t>& layers, 
                                                         const std::string& activation, 
                                                         const std::string& loss, 
                                                         T learningRate)
    : NeuralNetwork<T>(layers, activation, loss, learningRate) {
}

// Constructor from file
template <typename T>
LinearOutputNeuralNetwork<T>::LinearOutputNeuralNetwork(const std::string& filename)
    : NeuralNetwork<T>(filename) {
}

// Forward propagation with linear output layer
template <typename T>
Vector<T> LinearOutputNeuralNetwork<T>::forward(const Vector<T>& input, size_t layerIndex) {
    if (input.getSize() != this->m_layers[layerIndex]) {
        std::cerr << "Error: Input size (" << input.getSize() 
                  << ") does not match layer " << layerIndex << " size (" 
                  << this->m_layers[layerIndex] << ")" << std::endl;
        return Vector<T>(this->m_layers.back());
    }
    
    // First activation is the input itself
    this->m_activations[0] = input;
    
    Vector<T> currentActivation = input;
    
    // Propagate through all layers starting from layerIndex
    for (size_t layer = layerIndex; layer < this->m_weights.size(); ++layer) {
        // Compute z = W * a + b
        Vector<T> z = this->m_weights[layer] * currentActivation + this->m_biases[layer];;
        
        this->m_preActivations[layer] = z;
        
        // Check if this is the last layer
        bool isLastLayer = (layer == this->m_weights.size() - 1);
        
        if (isLastLayer) {
            // Linear output: no activation function
            currentActivation = z;
        } else {
            // Apply activation function for hidden layers
            if (this->m_activation == "sigmoid")
                currentActivation = this->m_activationFunction.sigmoid(z);
            else if (this->m_activation == "tanh")
                currentActivation = this->m_activationFunction.tanh(z);
            else if (this->m_activation == "ReLU")
                currentActivation = this->m_activationFunction.ReLU(z);
            else if (this->m_activation == "LeakyReLU")
                currentActivation = this->m_activationFunction.LeakyReLU(z);
            else {
                std::cerr << "Error: Unknown activation function " << this->m_activation << std::endl;
                return Vector<T>(this->m_layers.back());
            }
        }
        this->m_activations[layer+1] = currentActivation;
    }
    
    return currentActivation;
}

// Backward propagation with linear output layer
template <typename T>
T LinearOutputNeuralNetwork<T>::backward(const Vector<T>& input, const Vector<T>& target) {
    // Forward pass
    Vector<T> output = forward(input);
    
    // Compute loss
    T loss;
    if (this->m_loss == "meanSquaredError")
        loss = this->m_lossFunction.meanSquaredError(output, target);
    else {
        std::cerr << "Error: Unknown loss function " << this->m_loss << std::endl;
        return T(0);
    }
    
    // Compute loss gradient with respect to output
    Vector<T> dLoss(this->m_layers.back());
    if (this->m_loss == "meanSquaredError") {
        dLoss = this->m_lossFunction.meanSquaredErrorDerivative(output, target);
    } else {
        std::cerr << "Error: Unknown loss function " << this->m_loss << std::endl;
        return T(0);
    }
    
    // Backpropagation
    
    // Output layer - linear activation, so derivative is 1
    // Delta of the last layer: dLoss * 1 = dLoss (no activation derivative)
    size_t lastLayerIdx = this->m_layers.size() - 1;
    size_t weightIdx = this->m_weights.size() - 1;
    Vector<T> delta = dLoss;  // Linear output: derivative = 1
    this->m_deltas[lastLayerIdx] = delta;
    
    // Backpropagation to hidden layers
    for (int i = lastLayerIdx - 1; i > 0; --i) {
        // delta[layer] = (W[layer+1]^T * delta[layer+1]) * activation'(z[layer])
        
        // Compute W^T * delta using transpose
        Vector<T> weightedDelta = this->m_weights[i].multiplyTranspose(this->m_deltas[i+1]);
        
        // Multiply by activation'(z) using Hadamard product
        Vector<T> deriv(this->m_layers[i]);
        if (this->m_activation == "sigmoid")
            deriv = this->m_activationFunction.sigmoidDerivative(this->m_preActivations[i-1]);
        else if (this->m_activation == "tanh")
            deriv = this->m_activationFunction.tanhDerivative(this->m_preActivations[i-1]);
        else if (this->m_activation == "ReLU")
            deriv = this->m_activationFunction.ReLUDerivative(this->m_preActivations[i-1]);
        else if (this->m_activation == "LeakyReLU")
            deriv = this->m_activationFunction.LeakyReLUDerivative(this->m_preActivations[i-1]);
        else {
            std::cerr << "Error: Unknown activation function " << this->m_activation << std::endl;
        }

        this->m_deltas[i] = weightedDelta.hadamard(deriv);
    }
    
    // Update weights and biases
    for (size_t layer = 0; layer < this->m_weights.size(); ++layer) {
        // Update weights: W -= learningRate * (delta * a^T)
        this->m_weights[layer].rank1Update(this->m_deltas[layer+1], this->m_activations[layer], this->m_learningRate);
        
        // Update biases: b -= learningRate * delta
        this->m_biases[layer] -= (this->m_deltas[layer+1] * this->m_learningRate);
    }
    
    return loss;
}

// Explicit template instantiation for common types
template class NeuralNetwork<float>;
template class NeuralNetwork<double>;
template class LinearOutputNeuralNetwork<float>;
template class LinearOutputNeuralNetwork<double>;
