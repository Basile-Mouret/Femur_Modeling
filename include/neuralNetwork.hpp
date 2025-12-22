#ifndef NEURAL_NETWORK_HPP
#define NEURAL_NETWORK_HPP

#include <iostream>
#include <vector>
#include <string>
#include "linalg.hpp"


class NeuralNetwork{
    protected:
        std::vector<Matrix2D<float>> m_weights;
        std::vector<Vector<float>> m_biases;
        std::vector<size_t> m_layer_sizes;
        std::string m_activation_function;
        std::string m_loss_function;
        
    public:
        // Constructors

        /*
        * @brief Construct a NeuralNetwork with given layer sizes, activation function and loss function
        * @param layer_sizes: const std::vector<size_t>&
        * @param activation_function: const std::string&
        * @param loss_function: const std::string&
        */
        NeuralNetwork(const std::vector<size_t>& layer_sizes, const std::string& activation_function, const std::string& loss_function);

        /*
        * @brief Construct a NeuralNetwork by loading model from file
        * @param model_filename: const std::string&
        */
        NeuralNetwork(const std::string& model_filename);

        /*
        * @brief Initialize weights and biases with random values
        */
        void initializeWeightsAndBiases();

        /*        
        * @brief Perform forward propagation through the network. Write the list of activations for each layer and the list of the value before activation (z) for each layer.
        * @param input: const Vector<float>&
        * @param activations: std::vector<Vector<float>>&
        * @param zs: std::vector<Vector<float>>&
        */
        void forwardPropagation(const Vector<float>& input, std::vector<Vector<float>>* activations, std::vector<Vector<float>>* z_values);

        /*
        * @brief Perform backward propagation through the network. Update the weights and biases based on the provided inputs and targets. Store the activations and z values for each layer.
        * @param inputs: const std::vector<Vector<float>>&
        * @param targets: const std::vector<Vector<float>>&
        * @param activations: std::vector<Matrix2D<float>>*
        * @param z_values: std::vector<Vector<float>>*
        */
        void backwardPropagation(const std::vector<Vector<float>>& inputs, 
                                const std::vector<Vector<float>>& targets,
                                std::vector<Matrix2D<float>>* activations,
                                std::vector<Vector<float>>* z_values);

        /*
        * @brief Train the neural network using the provided training data for a specified number of epochs and learning rate.
        * @param training_inputs: const std::vector<Vector<float>>&
        * @param training_targets: const std::vector<Vector<float>>&
        * @param epochs: size_t
        * @param learning_rate: float
        */
        void train(const std::vector<Vector<float>>& training_inputs, 
                    const std::vector<Vector<float>>& training_targets,
                    size_t epochs,
                    float learning_rate);

        /*
        * @brief Predict the output for a given input vector.
        * @param input: const Vector<float>&
        * @return Vector<float>
        */
        Vector<float> predict(const Vector<float>& input);
        
        /*
        * @brief Save the model weights and biases to a file.   
        * @param filename: const std::string&
        */
        void saveModel(const std::string& filename);       

} 

#endif