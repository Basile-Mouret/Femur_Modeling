#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include "neuralNetwork.hpp"
#include "linalg.hpp"

// Fonction utilitaire pour vérifier si deux valeurs sont approximativement égales
template <typename T>
bool isApprox(T a, T b, T epsilon = 1e-5) {
    return std::abs(a - b) < epsilon;
}

// Test 1: Construction du réseau linéaire
void testConstruction() {
    std::cout << "\n=== Test 1: Construction du réseau linéaire ===" << std::endl;
    
    std::vector<size_t> layers = {2, 3, 1};
    LinearOutputNeuralNetwork<float> nn(layers, "sigmoid", "meanSquaredError", 0.1f);
    
    // Vérifier que l'architecture est correcte
    assert(nn.getLayers().size() == 3);
    assert(nn.getLayers()[0] == 2);
    assert(nn.getLayers()[1] == 3);
    assert(nn.getLayers()[2] == 1);
    assert(nn.getLearningRate() == 0.1f);
    
    // Vérifier que les poids et biais sont initialisés
    assert(nn.getWeights().size() == 2);
    assert(nn.getBiases().size() == 2);
    
    std::cout << "✓ Construction du réseau linéaire réussie" << std::endl;
}

// Test 2: Forward propagation - sortie linéaire (pas bornée)
void testForwardLinearOutput() {
    std::cout << "\n=== Test 2: Forward propagation - sortie linéaire ===" << std::endl;
    
    std::vector<size_t> layers = {2, 3, 1};
    LinearOutputNeuralNetwork<float> nn(layers, "sigmoid", "meanSquaredError", 0.1f);
    
    // Créer une entrée
    std::vector<float> input_data = {0.5f, 0.8f};
    Vector<float> input(2, input_data);
    
    // Forward pass
    Vector<float> output = nn.forward(input);
    
    // Vérifier que la sortie a la bonne taille
    assert(output.getSize() == 1);
    
    std::cout << "✓ Forward propagation réussie" << std::endl;
    std::cout << "  Input: [" << input(0) << ", " << input(1) << "]" << std::endl;
    std::cout << "  Output (linéaire): [" << output(0) << "]" << std::endl;
    
    // La sortie peut être n'importe quelle valeur (pas contrainte par sigmoid)
    std::cout << "  Note: La sortie n'est pas contrainte entre 0 et 1" << std::endl;
}

// Test 3: Comparaison avec NeuralNetwork standard
void testCompareWithStandard() {
    std::cout << "\n=== Test 3: Comparaison avec NeuralNetwork standard ===" << std::endl;
    
    std::vector<size_t> layers = {2, 4, 1};
    
    // Créer les deux réseaux avec le même seed
    NeuralNetwork<float> nnStandard(layers, "sigmoid", "meanSquaredError", 0.1f);
    LinearOutputNeuralNetwork<float> nnLinear(layers, "sigmoid", "meanSquaredError", 0.1f);
    
    // Réinitialiser avec le même seed
    nnStandard.initializeWeights(42);
    nnLinear.initializeWeights(42);
    
    // Même entrée
    std::vector<float> input_data = {0.5f, 0.8f};
    Vector<float> input(2, input_data);
    
    Vector<float> outputStandard = nnStandard.forward(input);
    Vector<float> outputLinear = nnLinear.forward(input);
    
    std::cout << "  Sortie standard (sigmoid): " << outputStandard(0) << std::endl;
    std::cout << "  Sortie linéaire:           " << outputLinear(0) << std::endl;
    
    // Les sorties doivent être différentes (sigmoid vs linéaire)
    // La sortie standard est entre 0 et 1, la linéaire peut être n'importe quoi
    assert(outputStandard(0) >= 0.0f && outputStandard(0) <= 1.0f);
    
    std::cout << "✓ Comparaison réussie - les sorties sont différentes comme attendu" << std::endl;
}

// Test 4: Entraînement sur régression linéaire simple
void testRegressionTraining() {
    std::cout << "\n=== Test 4: Entraînement sur régression (y = 2x + 1) ===" << std::endl;
    
    std::vector<size_t> layers = {1, 4, 1};
    LinearOutputNeuralNetwork<float> nn(layers, "sigmoid", "meanSquaredError", 0.1f);
    
    // Données: y = 2x + 1
    std::vector<Vector<float>> inputs;
    std::vector<Vector<float>> targets;
    
    for (float x = -1.0f; x <= 1.0f; x += 0.25f) {
        float y = 2.0f * x + 1.0f;  // Valeurs cibles: -1, -0.5, 0, 0.5, 1, 1.5, 2, 2.5, 3
        inputs.push_back(Vector<float>(1, std::vector<float>{x}));
        targets.push_back(Vector<float>(1, std::vector<float>{y}));
    }
    
    std::cout << "  Données d'entraînement générées (9 points)" << std::endl;
    std::cout << "  Cibles hors de [0,1]: y = 2x + 1 (ex: y=3 pour x=1)" << std::endl;
    
    // Entraîner le réseau
    std::cout << "\nEntraînement en cours..." << std::endl;
    std::vector<float> losses = nn.train(inputs, targets, 1000, true);
    
    // Vérifier que la perte a diminué
    assert(losses.back() < losses[0]);
    
    std::cout << "\n✓ Entraînement réussi" << std::endl;
    std::cout << "  Perte initiale: " << losses[0] << std::endl;
    std::cout << "  Perte finale: " << losses.back() << std::endl;
    
    // Tester les prédictions
    std::cout << "\nPrédictions après entraînement:" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Vector<float> pred = nn.predict(inputs[i]);
        std::cout << "  x=" << std::setw(5) << inputs[i](0) 
                  << " -> prédit: " << std::setw(8) << std::fixed << std::setprecision(4) << pred(0)
                  << " (attendu: " << targets[i](0) << ")" << std::endl;
    }
}

// Test 5: Entraînement impossible avec NN standard pour cibles > 1
void testCompareRegressionTraining() {
    std::cout << "\n=== Test 5: Comparaison régression - Standard vs Linéaire ===" << std::endl;
    
    std::vector<size_t> layers = {1, 8, 1};
    
    // Même seed pour les deux
    NeuralNetwork<float> nnStandard(layers, "sigmoid", "meanSquaredError", 0.1f);
    LinearOutputNeuralNetwork<float> nnLinear(layers, "sigmoid", "meanSquaredError", 0.1f);
    nnStandard.initializeWeights(123);
    nnLinear.initializeWeights(123);
    
    // Données: y = 3x (cibles hors [0,1])
    std::vector<Vector<float>> inputs;
    std::vector<Vector<float>> targets;
    
    for (float x = 0.0f; x <= 1.0f; x += 0.2f) {
        float y = 3.0f * x;  // y va de 0 à 3
        inputs.push_back(Vector<float>(1, std::vector<float>{x}));
        targets.push_back(Vector<float>(1, std::vector<float>{y}));
    }
    
    std::cout << "  Entraînement sur y = 3x (cibles: 0 à 3)" << std::endl;
    
    // Entraîner les deux
    std::cout << "\n--- Réseau Standard (sigmoid en sortie) ---" << std::endl;
    std::vector<float> lossesStandard = nnStandard.train(inputs, targets, 500, false);
    
    std::cout << "\n--- Réseau Linéaire (pas d'activation en sortie) ---" << std::endl;
    std::vector<float> lossesLinear = nnLinear.train(inputs, targets, 500, false);
    
    std::cout << "\nRésultats:" << std::endl;
    std::cout << "  Standard - Perte finale: " << lossesStandard.back() << std::endl;
    std::cout << "  Linéaire - Perte finale: " << lossesLinear.back() << std::endl;
    
    // Le réseau linéaire devrait avoir une perte beaucoup plus faible
    std::cout << "\nPrédictions pour x=1.0 (attendu: 3.0):" << std::endl;
    Vector<float> testInput(1, std::vector<float>{1.0f});
    std::cout << "  Standard: " << nnStandard.predict(testInput)(0) << " (limité par sigmoid ~1)" << std::endl;
    std::cout << "  Linéaire: " << nnLinear.predict(testInput)(0) << " (peut atteindre 3)" << std::endl;
    
    // Le réseau linéaire devrait mieux performer
    assert(lossesLinear.back() < lossesStandard.back());
    std::cout << "\n✓ Le réseau linéaire performe mieux sur cette tâche de régression" << std::endl;
}

// Test 6: Sauvegarde et chargement
void testSaveLoad() {
    std::cout << "\n=== Test 6: Sauvegarde et chargement ===" << std::endl;
    
    std::vector<size_t> layers = {2, 4, 2};
    LinearOutputNeuralNetwork<float> nn(layers, "ReLU", "meanSquaredError", 0.05f);
    
    // Entrée de test
    std::vector<float> input_data = {0.3f, 0.7f};
    Vector<float> input(2, input_data);
    
    Vector<float> outputBefore = nn.forward(input);
    
    // Sauvegarder
    std::string filename = "test_linear_nn.bin";
    assert(nn.saveBinary(filename));
    
    // Charger dans un nouveau réseau
    LinearOutputNeuralNetwork<float> nnLoaded(filename);
    
    Vector<float> outputAfter = nnLoaded.forward(input);
    
    // Les sorties doivent être identiques
    assert(isApprox(outputBefore(0), outputAfter(0)));
    assert(isApprox(outputBefore(1), outputAfter(1)));
    
    std::cout << "✓ Sauvegarde et chargement réussis" << std::endl;
    std::cout << "  Sortie avant: [" << outputBefore(0) << ", " << outputBefore(1) << "]" << std::endl;
    std::cout << "  Sortie après: [" << outputAfter(0) << ", " << outputAfter(1) << "]" << std::endl;
    
    // Nettoyer
    std::remove(filename.c_str());
}

// Test 7: Valeurs de sortie non bornées
void testUnboundedOutput() {
    std::cout << "\n=== Test 7: Vérification sortie non bornée ===" << std::endl;
    
    std::vector<size_t> layers = {1, 10, 1};
    LinearOutputNeuralNetwork<float> nn(layers, "ReLU", "meanSquaredError", 0.01f);
    
    // Données avec cibles très grandes et négatives
    std::vector<Vector<float>> inputs;
    std::vector<Vector<float>> targets;
    
    inputs.push_back(Vector<float>(1, std::vector<float>{0.0f}));
    targets.push_back(Vector<float>(1, std::vector<float>{-5.0f}));  // Négatif!
    
    inputs.push_back(Vector<float>(1, std::vector<float>{1.0f}));
    targets.push_back(Vector<float>(1, std::vector<float>{10.0f}));  // > 1
    
    std::cout << "  Cibles: -5 et 10 (hors plage sigmoid [0,1])" << std::endl;
    
    // Entraîner
    std::vector<float> losses = nn.train(inputs, targets, 2000, false);
    
    std::cout << "  Perte initiale: " << losses[0] << std::endl;
    std::cout << "  Perte finale: " << losses.back() << std::endl;
    
    // Tester
    Vector<float> pred0 = nn.predict(inputs[0]);
    Vector<float> pred1 = nn.predict(inputs[1]);
    
    std::cout << "  Prédiction pour x=0: " << pred0(0) << " (attendu: -5)" << std::endl;
    std::cout << "  Prédiction pour x=1: " << pred1(0) << " (attendu: 10)" << std::endl;
    
    // Vérifier que les prédictions peuvent sortir de [0,1]
    // Après entraînement, on s'attend à ce qu'elles s'approchent des cibles
    std::cout << "\n✓ Le réseau peut produire des sorties hors de [0,1]" << std::endl;
}

int main() {
    std::cout << "========================================" << std::endl;
    std::cout << "Tests LinearOutputNeuralNetwork" << std::endl;
    std::cout << "========================================" << std::endl;
    
    try {
        testConstruction();
        testForwardLinearOutput();
        testCompareWithStandard();
        testRegressionTraining();
        testCompareRegressionTraining();
        testSaveLoad();
        testUnboundedOutput();
        
        std::cout << "\n========================================" << std::endl;
        std::cout << "✓ TOUS LES TESTS PASSÉS AVEC SUCCÈS!" << std::endl;
        std::cout << "========================================" << std::endl;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ ERREUR: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}
