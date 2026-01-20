#include <iostream>
#include <filesystem>
#include <chrono>
#include <fstream>
#include "neuralNetwork.hpp"
#include "femur.hpp"

int main() {
    float maxDifference = 36.f;
    std::cout << "Femur Training Timing Test" << std::endl;

    std::vector<Vector<float>> training_data;
    std::cout << "Loading Femurs" << std::endl;
    Femur meanFemur("../data/mean_femur.obj");
    Vector<float> meanFemurCoords(meanFemur.getCoordsVect<float>());

    Femur femur;
    std::string trainingFolderPath = "../data/training";
    for (const auto& entry : std::filesystem::directory_iterator(trainingFolderPath)) {
        femur = Femur(entry.path());
        training_data.push_back((femur.getCoordsVect<float>()-meanFemurCoords)*(1.f/maxDifference));
    }

    std::cout << "Loading Neural Network" << std::endl;
    std::vector<size_t> layers = {54873, 512, 64, 10, 64, 512, 54873};
    LinearOutputNeuralNetwork<float> nn(layers, "LeakyReLU", "meanSquaredError", 10.f);
    nn.loadBinary("../models/NeuralNetwork_centered_LReLU.bin");
    nn.setLearningRate(1.f);

    std::ofstream timing_file("epoch_times.txt");
    if (!timing_file.is_open()) {
        std::cerr << "Failed to open epoch_times.txt for writing." << std::endl;
        return 1;
    }

    std::cout << "\nTraining the Neural Network for 20 epochs..." << std::endl;
    int n_epochs = 20;
    for (int epoch = 0; epoch < n_epochs; ++epoch) {
        auto start = std::chrono::high_resolution_clock::now();
        nn.train(training_data, training_data, 1, false); // 1 epoch, no verbose
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> diff = end - start;
        timing_file << diff.count() << std::endl;
        std::cout << "Epoch " << (epoch+1) << "/" << n_epochs << ": " << diff.count() << " seconds." << std::endl;
    }
    timing_file.close();
    std::cout << "\nTiming data saved to epoch_times.txt" << std::endl;
    return 0;
}
