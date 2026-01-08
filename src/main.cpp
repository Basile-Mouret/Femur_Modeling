#include <iostream>
#include <filesystem>
#include "neuralNetwork.hpp"
#include "femur.hpp"

int main() {
    std::cout << "Femur Modeling Project" << std::endl;

    std::vector<Vector<float>> training_data;
    std::vector<Vector<float>> test_data;


    std::cout << "Loading Femurs" << std::endl;
    std::string trainingFolderPath = "../data/training";

    Femur femur;
    for (const auto& entry : std::filesystem::directory_iterator(trainingFolderPath)) {
        femur = Femur(entry.path());
        Vector<float> femurCoords = femur.getCoordsVect<float>();
        training_data.push_back(femurCoords);
    }

    std::cout << "Initializing Neural Network" << std::endl;
    std::vector<size_t> layers = {18291*3, 500, 10, 500, 18291*3};
    NeuralNetwork<float> nn(layers, 1.f);
    
    // Training NN
    std::cout << "\nTraining the Neural Network..." << std::endl;
    std::vector<float> losses = nn.train(training_data, training_data, 2, true);
    
    std::cout << "\n✓ Training Complete" << std::endl;
    std::cout << "  Initial loss : " << losses[0] << std::endl;
    std::cout << "  Final loss : " << losses.back() << std::endl;

    nn.save("NeuralNetwork.nn");

    return 0;
}

