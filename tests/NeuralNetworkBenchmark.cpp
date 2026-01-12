#include <random>
#include <algorithm>
#include "neuralNetwork.hpp"

int main() {
    std::cout << "Neural Network Benchmark - Modulo Addition" << std::endl;

    // Generate all data for (a + b) mod 5
    std::vector<Vector<double>> allInputs;
    std::vector<Vector<double>> allOutputs;
    long unsigned int k = 5;

    for (int a = 0; a < k; ++a) {
        for (int b = 0; b < k; ++b) {
            Vector<double> input(2*k);
            Vector<double> output(k);

            for (int i = 0; i < 2*k; ++i) input(i) = 0.0;
            input(a) = 1.0;
            input(k + b) = 1.0;

            int result = (a + b) % k;
            for (int i = 0; i < k; ++i) output(i) = 0.0;
            output(result) = 1.0;

            allInputs.push_back(input);
            allOutputs.push_back(output);
        }
    }

    // Shuffle data
    std::vector<size_t> indices(allInputs.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::mt19937 rng(42);
    std::shuffle(indices.begin(), indices.end(), rng);

    // Split: 80% training, 20% validation
    size_t trainSize = static_cast<size_t>(allInputs.size() * 0.8);

    std::vector<Vector<double>> trainInputs, trainOutputs;
    std::vector<Vector<double>> valInputs, valOutputs;

    for (size_t i = 0; i < indices.size(); ++i) {
        if (i < trainSize) {
            trainInputs.push_back(allInputs[indices[i]]);
            trainOutputs.push_back(allOutputs[indices[i]]);
        } else {
            valInputs.push_back(allInputs[indices[i]]);
            valOutputs.push_back(allOutputs[indices[i]]);
        }
    }

    std::cout << "Training samples: " << trainInputs.size() << std::endl;
    std::cout << "Validation samples: " << valInputs.size() << std::endl;

    // Small network: 10 -> 16 -> 16 -> 5
    std::vector<size_t> layers = {2*k, 3*k, k};
    NeuralNetwork<double> nn(layers, 0.5);

    std::cout << "\nTraining the Neural Network..." << std::endl;
    std::vector<double> losses = nn.train(trainInputs, trainOutputs, 5000, true);

    std::cout << "\n✓ Training Complete" << std::endl;
    std::cout << "  Initial loss: " << losses[0] << std::endl;
    std::cout << "  Final loss: " << losses.back() << std::endl;

    // Evaluate on validation set
    std::cout << "\nValidation Results:" << std::endl;
    int correct = 0;
    for (size_t i = 0; i < valInputs.size(); ++i) {
        Vector<double> output = nn.forward(valInputs[i]);
        int predicted = 0, expected = 0;
        for (int j = 1; j < k; ++j) {
            if (output(j) > output(predicted)) predicted = j;
            if (valOutputs[i](j) > valOutputs[i](expected)) expected = j;
        }
        float eps=1e-6;
        if (expected-eps < predicted < expected+eps) ++correct;
    }

    double accuracy = 100.0 * correct / valInputs.size();
    std::cout << "  Accuracy: " << correct << "/" << valInputs.size()
              << " (" << accuracy << "%)" << std::endl;

    return 0;
}
