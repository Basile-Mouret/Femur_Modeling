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
        Vector<float> femurCoordsStandardized = femur.getCoordsVect<float>(54, true);
        training_data.push_back(femurCoordsStandardized);
    }


    std::cout << "Initializing Neural Network" << std::endl;
    std::vector<size_t> layers = {1017, 512, 256, 128, 64, 64, 128, 256, 512, 1017};
    NeuralNetwork<float> nn(layers, .01f);
    
    // Training NN
    std::cout << "\nTraining the Neural Network..." << std::endl;
    std::vector<float> losses = nn.train(training_data, training_data, 1000, true);
    
    std::cout << "\n✓ Training Complete" << std::endl;
    std::cout << "  Initial loss : " << losses[0] << std::endl;
    std::cout << "  Final loss : " << losses.back() << std::endl;

    nn.save("NeuralNetwork.nn");

    return 0;
}

