/**
 * @file neuralNetworkFunctions.hpp
 * @brief Activation and loss functions for neural networks
 * @details This file contains template classes for activation functions (sigmoid)
 *          and loss functions (mean squared error) used in neural network training
 *          and inference. Supports both scalar and vector operations.
 */

#ifndef NEURAL_NETWORK_FUNCTIONS_HPP
#define NEURAL_NETWORK_FUNCTIONS_HPP



#include <vector>
#include "linalg.hpp"

/**
 * @class ActivationFunction
 * @brief Template class providing activation functions for neural networks
 * @tparam T Data type for computations (typically float or double)
 * 
 * This class implements the sigmoid activation function and its derivative,
 * supporting operations on scalars, vectors, and collections of vectors.
 * The sigmoid function maps any input to the range (0, 1).
 */
template <typename T>
class ActivationFunction {
    public:
        // Sigmoid activation function and its derivative

        /**
         * @brief Applies the sigmoid activation function to a scalar
         * 
         * Computes σ(x) = 1 / (1 + e^(-x)), mapping the input to (0, 1).
         * 
         * @param x Input scalar value
         * @return Activated value in range (0, 1)
         */
        T sigmoid(T x);

        /**
         * @brief Applies the derivative of the sigmoid function to a scalar
         * 
         * Computes σ'(x) = σ(x) * (1 - σ(x)), used for backpropagation.
         * 
         * @param x Input scalar value
         * @return Derivative value
         */
        T sigmoidDerivative(T x);

        /**
         * @brief Applies the sigmoid activation function to a vector
         * 
         * Applies sigmoid element-wise to each component of the input vector.
         * 
         * @param vec Input vector
         * @return Vector with sigmoid applied to each element
         */
        Vector<T> sigmoid(const Vector<T>& vec);

        /**
         * @brief Applies the derivative of sigmoid to a vector
         * 
         * Applies sigmoid derivative element-wise to each component of the input vector.
         * 
         * @param vec Input vector
         * @return Vector with sigmoid derivative applied to each element
         */
        Vector<T> sigmoidDerivative(const Vector<T>& vec);

        /**
         * @brief Applies sigmoid to a collection of vectors
         * 
         * Applies sigmoid function to each vector in the collection.
         * 
         * @param vecs Collection of input vectors
         * @return Collection of vectors with sigmoid applied
         */
        std::vector<Vector<T>> sigmoid(const std::vector<Vector<T>>& vecs);

        /**
         * @brief Applies sigmoid derivative to a collection of vectors
         * 
         * Applies sigmoid derivative to each vector in the collection.
         * 
         * @param vecs Collection of input vectors
         * @return Collection of vectors with sigmoid derivative applied
         */
        std::vector<Vector<T>> sigmoidDerivative(const std::vector<Vector<T>>& vecs);
};

/**
 * @class LossFunction
 * @brief Template class providing loss functions for neural networks
 * @tparam T Data type for computations (typically float or double)
 * 
 * This class implements the Mean Squared Error (MSE) loss function and
 * its derivative, commonly used for regression tasks. MSE measures the
 * average squared difference between predictions and targets.
 */
template <typename T>
class LossFunction {
    public:
        // Mean Squared Error (MSE) loss function and its derivative

        /**
         * @brief Computes the Mean Squared Error loss
         * 
         * Calculates MSE = (1/n) * Σ(predicted_i - target_i)²
         * where n is the vector size.
         * 
         * @param predicted Network prediction vector
         * @param target Ground truth target vector
         * @return Scalar MSE loss value
         */
        T meanSquaredError(const Vector<T>& predicted, const Vector<T>& target);

        /**
         * @brief Computes the derivative of the Mean Squared Error loss
         * 
         * Calculates ∂MSE/∂predicted = (2/n) * (predicted - target)
         * for use in backpropagation.
         * 
         * @param predicted Network prediction vector
         * @param target Ground truth target vector
         * @return Vector of partial derivatives with respect to predictions
         */
        Vector<T> meanSquaredErrorDerivative(const Vector<T>& predicted, const Vector<T>& target);
};

#endif // NEURAL_NETWORK_FUNCTIONS_HPP