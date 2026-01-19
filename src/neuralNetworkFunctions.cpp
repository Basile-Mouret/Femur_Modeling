#include "neuralNetworkFunctions.hpp"
#include <math.h>

// Sigmoid activation function and its derivative

template <typename T>
T ActivationFunction<T>::sigmoid(T x) {
    return static_cast<T>(1) / (static_cast<T>(1) + exp(-x));
}

template <typename T>
T ActivationFunction<T>::sigmoidDerivative(T x) {
    T sig = sigmoid(x);
    return sig * (static_cast<T>(1) - sig);
}


template <typename T>
Vector<T> ActivationFunction<T>::sigmoid(const Vector<T>& vec) {
    Vector<T> result(vec.getSize());
    for (size_t i = 0; i < vec.getSize(); ++i) {
        result.setCoeff(i, sigmoid(vec(i)));
    }
    return result;
}

template <typename T>
Vector<T> ActivationFunction<T>::sigmoidDerivative(const Vector<T>& vec) {
    Vector<T> result(vec.getSize());
    for (size_t i = 0; i < vec.getSize(); ++i) {
        result.setCoeff(i, sigmoidDerivative(vec(i)));
    }
    return result;
}

template <typename T>
std::vector<Vector<T>> ActivationFunction<T>::sigmoid(const std::vector<Vector<T>>& vecs) {
    std::vector<Vector<T>> result;
    for (const auto& vec : vecs) {
        result.push_back(sigmoid(vec));
    }
    return result;
}

template <typename T>
std::vector<Vector<T>> ActivationFunction<T>::sigmoidDerivative(const std::vector<Vector<T>>& vecs) {
    std::vector<Vector<T>> result;
    for (const auto& vec : vecs) {
        result.push_back(sigmoidDerivative(vec));
    }
    return result;
}

template <typename T>
T ActivationFunction<T>::tanh(T x) {
    return std::tanh(x);
}

template <typename T>
T ActivationFunction<T>::tanhDerivative(T x) {
    T t = tanh(x);
    return static_cast<T>(1) - t * t;
}

template <typename T>
Vector<T> ActivationFunction<T>::tanh(const Vector<T>& vec) {
    Vector<T> result(vec.getSize());
    for (size_t i = 0; i < vec.getSize(); ++i) {
        result.setCoeff(i, tanh(vec(i)));
    }
    return result;
}

template <typename T>
Vector<T> ActivationFunction<T>::tanhDerivative(const Vector<T>& vec) {
    Vector<T> result(vec.getSize());
    for (size_t i = 0; i < vec.getSize(); ++i) {
        result.setCoeff(i, tanhDerivative(vec(i)));
    }
    return result;
}

template <typename T>
std::vector<Vector<T>> ActivationFunction<T>::tanh(const std::vector<Vector<T>>& vecs) {
    std::vector<Vector<T>> result;
    for (const auto& vec : vecs) {
        result.push_back(tanh(vec));
    }
    return result;
}

template <typename T>
std::vector<Vector<T>> ActivationFunction<T>::tanhDerivative(const std::vector<Vector<T>>& vecs) {
    std::vector<Vector<T>> result;
    for (const auto& vec : vecs) {
        result.push_back(tanhDerivative(vec));
    }
    return result;
}

template <typename T>
T ActivationFunction<T>::ReLU(T x) {
    return (x>0 ? x : static_cast<T>(0));
}

template <typename T>
T ActivationFunction<T>::ReLUDerivative(T x) {
    T t = ReLU(x);
    return (x>0 ? static_cast<T>(1) : static_cast<T>(0));
}

template <typename T>
Vector<T> ActivationFunction<T>::ReLU(const Vector<T>& vec) {
    Vector<T> result(vec.getSize());
    for (size_t i = 0; i < vec.getSize(); ++i) {
        result.setCoeff(i, ReLU(vec(i)));
    }
    return result;
}

template <typename T>
Vector<T> ActivationFunction<T>::ReLUDerivative(const Vector<T>& vec) {
    Vector<T> result(vec.getSize());
    for (size_t i = 0; i < vec.getSize(); ++i) {
        result.setCoeff(i, ReLUDerivative(vec(i)));
    }
    return result;
}

template <typename T>
std::vector<Vector<T>> ActivationFunction<T>::ReLU(const std::vector<Vector<T>>& vecs) {
    std::vector<Vector<T>> result;
    for (const auto& vec : vecs) {
        result.push_back(ReLU(vec));
    }
    return result;
}

template <typename T>
std::vector<Vector<T>> ActivationFunction<T>::ReLUDerivative(const std::vector<Vector<T>>& vecs) {
    std::vector<Vector<T>> result;
    for (const auto& vec : vecs) {
        result.push_back(ReLUDerivative(vec));
    }
    return result;
}

template <typename T>
T LossFunction<T>::meanSquaredError(const Vector<T>& predicted, const Vector<T>& actual) {
    if (predicted.getSize() != actual.getSize()) {
        std::cout << "Vectors must be of the same size." << std::endl;
        return static_cast<T>(0);
    }
    T sum = static_cast<T>(0);
    for (size_t i = 0; i < predicted.getSize(); ++i) {
        T diff = predicted(i) - actual(i);
        sum += diff * diff;
    }
    return sum / static_cast<T>(predicted.getSize());
}

template <typename T>
Vector<T> LossFunction<T>::meanSquaredErrorDerivative(const Vector<T>& predicted, const Vector<T>& actual) {
    if (predicted.getSize() != actual.getSize()) {
        std::cout << "Vectors must be of the same size." << std::endl;
        return Vector<T>(0);
    }
    Vector<T> result(predicted.getSize());
    for (size_t i = 0; i < predicted.getSize(); ++i) {
        result.setCoeff(i, static_cast<T>(2) * (predicted(i) - actual(i)) / static_cast<T>(predicted.getSize()));
    }
    return result;
}


// ============================================================================
// Explicit template instantiations
// ============================================================================

template class ActivationFunction<float>;
template class LossFunction<float>;

template class ActivationFunction<double>;
template class LossFunction<double>;

template class ActivationFunction<int>;
template class LossFunction<int>;

template class ActivationFunction<long>;
template class LossFunction<long>;

template class ActivationFunction<short>;
template class LossFunction<short>;

template class ActivationFunction<unsigned int>;
template class LossFunction<unsigned int>;

template class ActivationFunction<unsigned long>;
template class LossFunction<unsigned long>;
