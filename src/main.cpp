#include "dataset.hpp"
#include "neuralNetworkFunctions.hpp"
#include <iostream>
#include <filesystem>
#include  <chrono>
#include "neuralNetwork.hpp"
#include "femur.hpp"

// Calcule la MSE d'un modèle sur un dossier de fémurs
float evaluate_model_on_folder(const std::string& modelPath, const std::string& femurFolder, const std::string& meanFemurPath, float maxDifference = 36.f) {
    // Charger le fémur moyen
    Femur meanFemur(meanFemurPath);
    Vector<float> meanFemurCoords(meanFemur.getCoordsVect<float>());

    // Charger le modèle
    std::vector<size_t> layers = {meanFemurCoords.getSize(), 512, 64, 10, 64, 512, meanFemurCoords.getSize()};
    LinearOutputNeuralNetwork<float> nn(layers, "tanh", "meanSquaredError", 1.f);
    nn.loadBinary(modelPath);

    // Parcourir le dossier de fémurs
    size_t count = 0;
    float mse_sum = 0.f;
    LossFunction<float> lossFn;
    for (const auto& entry : std::filesystem::directory_iterator(femurFolder)) {
        if (entry.is_regular_file() && entry.path().extension() == ".obj") {
            Femur femur(entry.path().string());
            Vector<float> gt = femur.getCoordsVect<float>();
            // Normalisation
            Vector<float> input = (gt - meanFemurCoords) * (1.f / maxDifference);
            // Prédiction
            Vector<float> output = nn.forward(input) * maxDifference + meanFemurCoords;
            // Calcul MSE
            float mse = lossFn.meanSquaredError(output, gt);
            std::cout << "[" << entry.path().filename() << "] MSE = " << mse << std::endl;
            mse_sum += mse;
            count++;
        }
    }
    if (count == 0) {
        std::cerr << "Aucun fémur trouvé dans le dossier." << std::endl;
        return -1.f;
    }
    float mse_avg = mse_sum / count;
    std::cout << "\nMSE MOYENNE sur " << count << " fémurs : " << mse_avg << std::endl;
    return mse_avg;
}



int main(int argc, char** argv) {
    // Exemple d'utilisation :
    // Remplace les chemins par ceux de ton modèle, dossier de validation et fémur moyen
    std::string modelPath = argc > 1 ? argv[1] : "../models/NeuralNetwork.bin";
    std::string trainingFolder = "../data/training";
    std::string validationFolder = "../data/validation";
    std::string meanFemurPath = "../data/mean_femur.obj";
    evaluate_model_on_folder(modelPath, trainingFolder, meanFemurPath);
    evaluate_model_on_folder(modelPath, validationFolder, meanFemurPath);
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
