#ifndef NEURAL_NETWORK_FUNCTIONS_HPP
#define NEURAL_NETWORK_FUNCTIONS_HPP



#include <vector>
#include "linalg.hpp"

template <typename T>
class ActivationFunction {
    public:
        // Sigmoid activation function and its derivative

        /*
        * @brief Apply the sigmoid activation function.
        * @param x: Input value
        * @return Activated value
        */
        T sigmoid(T x);

        /*
        * @brief Apply the derivative of the sigmoid function.
        * @param x: Input value
        * @return Derivative value
        */
        T sigmoidDerivative(T x);

        /*
        * @brief Apply the sigmoid activation function.
        * @param vec: const Vector<T>&
        * @return Vector<T>
        */
        Vector<T> sigmoid(const Vector<T>& vec);

        /*
        * @brief Apply the derivative of the sigmoid function.
        * @param vec: const Vector<T>&
        * @return Vector<T>
        */
        Vector<T> sigmoidDerivative(const Vector<T>& vec);

        /*
        * @brief Apply the sigmoid activation function.
        * @param vecs: const std::vector<Vector<T>>&
        * @return std::vector<Vector<T>>
        */
        std::vector<Vector<T>> sigmoid(const std::vector<Vector<T>>& vecs);

        /*
        * @brief Apply the derivative of the sigmoid function.
        * @param vecs: const std::vector<Vector<T>>&
        * @return std::vector<Vector<T>>
        */
        std::vector<Vector<T>> sigmoidDerivative(const std::vector<Vector<T>>& vecs);
};

template <typename T>
class LossFunction {
    public:
        // Mean Squared Error (MSE) loss function and its derivative

        /*
        * @brief Compute the Mean Squared Error loss.
        * @param predicted: const Vector<float>&
        * @param target: const Vector<float>&
        * @return float
        */
        T meanSquaredError(const Vector<T>& predicted, const Vector<T>& target);

        /*
        * @brief Compute the derivative of the Mean Squared Error loss.
        * @param predicted: const Vector<float>&
        * @param target: const Vector<float>&
        * @return Vector<float>
        */
        Vector<T> meanSquaredErrorDerivative(const Vector<T>& predicted, const Vector<T>& target);
};
#endif // NEURAL_NETWORK_FUNCTIONS_HPP