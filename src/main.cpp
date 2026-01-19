#include <iostream>
#include <filesystem>
#include  <chrono>
#include "neuralNetwork.hpp"
#include "femur.hpp"


int main() {
    float maxDifference = 36.f;
    std::cout << "Femur Modeling Project" << std::endl;

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

    std::cout << "\nTraining the Neural Network..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now(); // Start
    std::vector<float> losses = nn.train(training_data, training_data, 100, true);
    auto end = std::chrono::high_resolution_clock::now();   // End
    std::chrono::duration<double> diff = end - start;
    std::cout << "\n✓ Training Complete in " << diff.count() << " seconds." << std::endl;

    if (nn.saveBinary("NeuralNetwork.bin"))
        std::cout << "Network saved successfully (binary)." << std::endl;
    else
        std::cerr << "Failed to save network (binary)." << std::endl;

    Femur reconstructedFemur("../data/validation/L_Femur_24_DECIM.obj.FINAL.obj");
    reconstructedFemur.setCoordsVect(nn.forward((reconstructedFemur.getCoordsVect<float>()-meanFemurCoords)*(1.f/maxDifference))*maxDifference+meanFemurCoords);
    reconstructedFemur.saveToFile("reconstructed_femur1.obj");

    Femur reconstructed2Femur("../data/validation/R_Femur_22_DECIM.obj.FINAL.obj");
    reconstructed2Femur.setCoordsVect(nn.forward((reconstructed2Femur.getCoordsVect<float>()-meanFemurCoords)*(1.f/maxDifference))*maxDifference+meanFemurCoords);
    reconstructed2Femur.saveToFile("reconstructed_femur2.obj");

    return 0;
}

/* new network

*/
/* old Neural Network
    std::vector<Vector<float>> training_data;

    std::cout << "Loading Femurs" << std::endl;
    std::string trainingFolderPath = "../data/training";

    Femur femur;
    for (const auto& entry : std::filesystem::directory_iterator(trainingFolderPath)) {
        femur = Femur(entry.path());
        Vector<float> femurCoordsStandardized = femur.getCoordsVect<float>(1, true);
        training_data.push_back(femurCoordsStandardized);
    }

    std::cout << "Initializing Neural Network" << std::endl;
    std::vector<size_t> layers = {54873, 1024, 64, 10, 64, 1024, 54873};
    NeuralNetwork<float> nn(layers, "tanh", "meanSquaredError", .1f);
    nn.loadBinary("../models/NeuralNetwork.bin");
    
    // Training NN
    //std::cout << "\nTraining the Neural Network..." << std::endl;
    //std::vector<float> losses = nn.train(training_data, training_data, 50, true);
    
    //std::cout << "\n✓ Training Complete" << std::endl;

    // Save in binary format for optimization
    if (nn.saveBinary("NeuralNetwork.bin")) {
        std::cout << "Network saved successfully (binary)." << std::endl;
    } else {
        std::cerr << "Failed to save network (binary)." << std::endl;
    }


    Femur reconstructedFemur("../data/validation/L_Femur_24_DECIM.obj.FINAL.obj");
    reconstructedFemur.setCoordsVect(nn.forward((reconstructedFemur.getCoordsVect<float>(1, true))));
    reconstructedFemur.saveToFile("reconstructed_femur1.obj");

    Femur reconstructed2Femur("../data/validation/R_Femur_22_DECIM.obj.FINAL.obj");
    reconstructed2Femur.setCoordsVect(nn.forward((reconstructed2Femur.getCoordsVect<float>(1, true))));
    reconstructed2Femur.saveToFile("reconstructed_femur2.obj");
 */
