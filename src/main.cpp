#include <iostream>
#include <filesystem>
#include "neuralNetwork.hpp"
#include "femur.hpp"

void smallFemur(){
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
    std::vector<size_t> layers = {1017, 512, 256, 128, 64, 32, 10, 32, 64, 128, 256, 512, 1017};
    NeuralNetwork<float> nn(layers, "sigmoid", "meanSquaredError", .01f);
    
    // Training NN
    std::cout << "\nTraining the Neural Network..." << std::endl;
    std::vector<float> losses = nn.train(training_data, training_data, 500, true);
    
    std::cout << "\n✓ Training Complete" << std::endl;
    std::cout << "  Initial loss : " << losses[0] << std::endl;
    std::cout << "  Final loss : " << losses.back() << std::endl;

    if (nn.saveBinary("NeuralNetwork.bin")) {
        std::cout << "Network saved successfully (binary)." << std::endl;
    } else {
        std::cerr << "Failed to save network (binary)." << std::endl;
    }
    


    Vector<float> result = nn.forward(training_data[0]);
    // Save the reconstructed femur
    Femur reconstructedFemur;
    reconstructedFemur.setCoordsVect(result);
    reconstructedFemur.saveToFile("reconstructed_femur.obj");

    Femur originalFemur;
    originalFemur.setCoordsVect(training_data[0]);
    originalFemur.saveToFile("original_data.obj");
}

void fullFemur(){
    std::vector<Vector<float>> training_data;
    std::vector<Vector<float>> test_data;


    std::cout << "Loading Femurs" << std::endl;
    std::string trainingFolderPath = "../data/training";

    Femur femur;
    for (const auto& entry : std::filesystem::directory_iterator(trainingFolderPath)) {
        femur = Femur(entry.path());
        Vector<float> femurCoordsStandardized = femur.getCoordsVect<float>();
        training_data.push_back(femurCoordsStandardized);
    }


    std::cout << "Initializing Neural Network" << std::endl;
    std::vector<size_t> layers = {54873, 1024, 64, 10, 64, 1024, 54873};
    NeuralNetwork<float> nn(layers, "tanh", "meanSquaredError", .1f);
    
    // Training NN
    std::cout << "\nTraining the Neural Network..." << std::endl;
    std::vector<float> losses = nn.train(training_data, training_data, 50, true);
    
    std::cout << "\n✓ Training Complete" << std::endl;
    std::cout << "  Initial loss : " << losses[0] << std::endl;
    std::cout << "  Final loss : " << losses.back() << std::endl;

    // Save in binary format for optimization
    if (nn.saveBinary("NeuralNetwork.bin")) {
        std::cout << "Network saved successfully (binary)." << std::endl;
    } else {
        std::cerr << "Failed to save network (binary)." << std::endl;
        // Fallback or error handling
    }
    
    // Also save in text format for compatibility if needed (optional)
    // nn.save("NeuralNetwork.nn");

    Vector<float> result = nn.forward(training_data[0]);
    // Save the reconstructed femur
    Femur reconstructedFemur;
    reconstructedFemur.setCoordsVect(result);
    reconstructedFemur.saveToFile("reconstructed_femur.obj");

    Femur originalFemur;
    originalFemur.setCoordsVect(training_data[0]);
    originalFemur.saveToFile("original_data.obj");
}

int main() {


    std::cout << "Femur Modeling Project" << std::endl;

    std::vector<Vector<float>> training_data_standardized;
    std::vector<Vector<float>> training_data_unstandardized;
    std::vector<Vector<float>> validation_data;


    std::cout << "Loading Femurs" << std::endl;
    std::string trainingFolderPath = "../data/training";
    std::string validationFolderPath = "../data/validation";

    Femur femur;
    for (const auto& entry : std::filesystem::directory_iterator(trainingFolderPath)) {
        femur = Femur(entry.path());
        training_data_standardized.push_back(femur.getCoordsVect<float>());
        training_data_unstandardized.push_back(femur.getCoordsVect<float>(1,false));
    }

    for (const auto& entry : std::filesystem::directory_iterator(validationFolderPath)) {
        femur = Femur(entry.path());
        Vector<float> femurCoordsStandardized = femur.getCoordsVect<float>();
        validation_data.push_back(femurCoordsStandardized);
    }
    std::cout << "Loading Neural Network" << std::endl;
    std::vector<size_t> layers = {54873, 512, 64, 10, 64, 512, 54873};
    LinearOutputNeuralNetwork<float> nn(layers, "ReLu", "meanSquaredError", .1f);

    
    // Training NN

    std::cout << "\nTraining the Neural Network..." << std::endl;
    std::vector<float> losses = nn.train(training_data_standardized, training_data_unstandardized, 30, true);
    
    std::cout << "\n✓ Training Complete" << std::endl;

    // Save in binary format for optimization
    if (nn.saveBinary("NeuralNetwork.bin")) {
        std::cout << "Network saved successfully (binary)." << std::endl;
    } else {
        std::cerr << "Failed to save network (binary)." << std::endl;
        // Fallback or error handling
    }

    // Save the reconstructed femur
    std::cout << validation_data.size() << std::endl;
    Femur reconstructedFemur("../data/validation/L_Femur_24_DECIM.obj.FINAL.obj");
    reconstructedFemur.setCoordsVect(nn.forward(validation_data[0]));
    reconstructedFemur.saveToFile("reconstructed_femur1.obj");

    // Save the reconstructed femur
    Femur reconstructed2Femur("../data/validation/R_Femur_22_DECIM.obj.FINAL.obj");
    reconstructed2Femur.setCoordsVect(nn.forward(validation_data[1]));
    reconstructed2Femur.saveToFile("reconstructed_femur2.obj");


    return 0;
}

