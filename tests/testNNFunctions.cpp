#include <iostream>
#include <vector>
#include <math.h>

#include "neuralNetworkFunctions.hpp"
#include "linalg.hpp"

int main(){
    ActivationFunction<double> af;
    LossFunction<double> lf;

    // Test sigmoid function
    double val = 0.0;
    double sig_val = af.sigmoid(val);
    std::cout << "Sigmoid(" << val << ") = " << sig_val << std::endl;

    // Test sigmoid derivative
    double sig_deriv = af.sigmoidDerivative(val);
    std::cout << "Sigmoid Derivative(" << val << ") = " << sig_deriv << std::endl;

    // Test Mean Squared Error
    Vector<double> predicted(3);
    predicted.setCoeff(0, 0.5);
    predicted.setCoeff(1, 0.6);
    predicted.setCoeff(2, 0.7);

    Vector<double> actual(3);
    actual.setCoeff(0, 0.4);
    actual.setCoeff(1, 0.6);
    actual.setCoeff(2, 0.8);

    double mse = lf.meanSquaredError(predicted, actual);
    std::cout << "Mean Squared Error = " << mse << std::endl;

    // Test Mean Squared Error Derivative
    Vector<double> mse_deriv = lf.meanSquaredErrorDerivative(predicted, actual);
    std::cout << "Mean Squared Error Derivative = ";
    for (size_t i = 0; i < mse_deriv.getSize(); ++i) {
        std::cout << mse_deriv(i) << " ";
    }
    std::cout << std::endl;

    // Test tanh function
    double tanh_val = af.tanh(val);
    std::cout << "Tanh(" << val << ") = " << tanh_val << std::endl;

    // Test tanh derivative
    double tanh_deriv = af.tanhDerivative(val);
    std::cout << "Tanh Derivative(" << val << ") = " << tanh_deriv << std::endl;

    return 0;
}