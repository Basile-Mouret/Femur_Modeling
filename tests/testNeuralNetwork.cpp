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

// Test 1: Construction du réseau
void testConstruction() {
    std::cout << "\n=== Test 1: Construction du réseau ===" << std::endl;
    
    std::vector<size_t> layers = {2, 3, 1};
    NeuralNetwork<float> nn(layers, 0.1f);
    
    // Vérifier que l'architecture est correcte
    assert(nn.getLayers().size() == 3);
    assert(nn.getLayers()[0] == 2);
    assert(nn.getLayers()[1] == 3);
    assert(nn.getLayers()[2] == 1);
    assert(nn.getLearningRate() == 0.1f);
    
    // Vérifier que les poids et biais sont initialisés
    assert(nn.getWeights().size() == 2);  // 2 matrices (input->hidden, hidden->output)
    assert(nn.getBiases().size() == 2);   // 2 vecteurs de biais
    
    std::cout << "✓ Construction du réseau réussie" << std::endl;
}

// Test 2: Forward propagation
void testForward() {
    std::cout << "\n=== Test 2: Forward propagation ===" << std::endl;
    
    std::vector<size_t> layers = {2, 2, 1};
    NeuralNetwork<float> nn(layers, 0.1f);
    
    // Créer une entrée
    std::vector<float> input_data = {0.5f, 0.8f};
    Vector<float> input(2, input_data);
    
    // Forward pass
    Vector<float> output = nn.forward(input);
    
    // Vérifier que la sortie a la bonne taille
    assert(output.getSize() == 1);
    
    // La sortie doit être entre 0 et 1 (sigmoid)
    assert(output(0) >= 0.0f && output(0) <= 1.0f);
    
    std::cout << "✓ Forward propagation réussie" << std::endl;
    std::cout << "  Input: [" << input(0) << ", " << input(1) << "]" << std::endl;
    std::cout << "  Output: [" << output(0) << "]" << std::endl;
}

// Test 3: Entraînement sur XOR
void testTrainingXOR() {
    std::cout << "\n=== Test 3: Entraînement sur XOR ===" << std::endl;
    
    // Architecture: 2 entrées, 4 neurones cachés, 1 sortie
    std::vector<size_t> layers = {2, 4, 1};
    NeuralNetwork<float> nn(layers, 0.5f);
    
    // Données XOR
    std::vector<Vector<float>> inputs;
    std::vector<Vector<float>> targets;
    
    // 0 XOR 0 = 0
    inputs.push_back(Vector<float>(2, std::vector<float>{0.0f, 0.0f}));
    targets.push_back(Vector<float>(1, std::vector<float>{0.0f}));
    
    // 0 XOR 1 = 1
    inputs.push_back(Vector<float>(2, std::vector<float>{0.0f, 1.0f}));
    targets.push_back(Vector<float>(1, std::vector<float>{1.0f}));
    
    // 1 XOR 0 = 1
    inputs.push_back(Vector<float>(2, std::vector<float>{1.0f, 0.0f}));
    targets.push_back(Vector<float>(1, std::vector<float>{1.0f}));
    
    // 1 XOR 1 = 0
    inputs.push_back(Vector<float>(2, std::vector<float>{1.0f, 1.0f}));
    targets.push_back(Vector<float>(1, std::vector<float>{0.0f}));
    
    // Entraîner le réseau
    std::cout << "\nEntraînement en cours..." << std::endl;
    std::vector<float> losses = nn.train(inputs, targets, 2000, true);
    
    // Vérifier que la perte a diminué
    assert(losses.back() < losses[0]);
    
    std::cout << "\n✓ Entraînement réussi" << std::endl;
    std::cout << "  Perte initiale: " << losses[0] << std::endl;
    std::cout << "  Perte finale: " << losses.back() << std::endl;
    
    // Tester les prédictions
    std::cout << "\nPrédictions après entraînement:" << std::endl;
    for (size_t i = 0; i < inputs.size(); ++i) {
        Vector<float> prediction = nn.predict(inputs[i]);
        std::cout << "  [" << inputs[i](0) << ", " << inputs[i](1) << "] -> " 
                  << std::fixed << std::setprecision(4) << prediction(0) 
                  << " (attendu: " << targets[i](0) << ")" << std::endl;
    }
}

// Test 4: Sauvegarde et chargement
void testSaveLoad() {
    std::cout << "\n=== Test 4: Sauvegarde et chargement ===" << std::endl;
    
    // Créer et entraîner un réseau
    std::vector<size_t> layers = {2, 3, 1};
    NeuralNetwork<float> nn1(layers, 0.3f);
    
    std::vector<Vector<float>> inputs;
    std::vector<Vector<float>> targets;
    
    inputs.push_back(Vector<float>(2, std::vector<float>{0.5f, 0.5f}));
    targets.push_back(Vector<float>(1, std::vector<float>{0.8f}));
    
    nn1.train(inputs, targets, 100, false);
    
    // Sauvegarder le réseau
    std::string filename = "test_network.txt";
    bool saved = nn1.save(filename);
    assert(saved);
    
    // Créer un nouveau réseau et charger
    std::vector<size_t> layers2 = {1, 1, 1};  // Architecture différente
    NeuralNetwork<float> nn2(filename);
    
    
    // Vérifier que l'architecture a été correctement chargée
    assert(nn2.getLayers().size() == nn1.getLayers().size());
    assert(nn2.getLayers()[0] == nn1.getLayers()[0]);
    assert(nn2.getLayers()[1] == nn1.getLayers()[1]);
    assert(nn2.getLayers()[2] == nn1.getLayers()[2]);
    assert(isApprox(nn2.getLearningRate(), nn1.getLearningRate()));
    
    // Vérifier que les prédictions sont identiques
    Vector<float> pred1 = nn1.predict(inputs[0]);
    Vector<float> pred2 = nn2.predict(inputs[0]);
    
    assert(isApprox(pred1(0), pred2(0), 1e-4f));
    
    std::cout << "✓ Sauvegarde et chargement réussis" << std::endl;
    std::cout << "  Prédiction réseau 1: " << pred1(0) << std::endl;
    std::cout << "  Prédiction réseau 2: " << pred2(0) << std::endl;
    
    // Nettoyer le fichier de test
    std::remove(filename.c_str());
}

// Test 5: Modification du taux d'apprentissage
void testLearningRateModification() {
    std::cout << "\n=== Test 5: Modification du taux d'apprentissage ===" << std::endl;
    
    std::vector<size_t> layers = {2, 2, 1};
    NeuralNetwork<float> nn(layers, 0.1f);
    
    assert(nn.getLearningRate() == 0.1f);
    
    nn.setLearningRate(0.5f);
    assert(nn.getLearningRate() == 0.5f);
    
    std::cout << "✓ Modification du taux d'apprentissage réussie" << std::endl;
}

// Test 6: Entraînement sur une fonction simple (approximation de y = x1 + x2)
void testSimpleFunction() {
    std::cout << "\n=== Test 6: Approximation de fonction (y = x1 + x2) ===" << std::endl;
    
    std::vector<size_t> layers = {2, 5, 1};
    NeuralNetwork<float> nn(layers, 0.3f);
    
    // Générer des données d'entraînement
    std::vector<Vector<float>> inputs;
    std::vector<Vector<float>> targets;
    
    for (float x1 = 0.0f; x1 <= 1.0f; x1 += 0.25f) {
        for (float x2 = 0.0f; x2 <= 1.0f; x2 += 0.25f) {
            inputs.push_back(Vector<float>(2, std::vector<float>{x1, x2}));
            // Normaliser la sortie entre 0 et 1: y = (x1 + x2) / 2
            float y = (x1 + x2) / 2.0f;
            targets.push_back(Vector<float>(1, std::vector<float>{y}));
        }
    }
    
    std::cout << "\nEntraînement sur " << inputs.size() << " exemples..." << std::endl;
    std::vector<float> losses = nn.train(inputs, targets, 1000, true);
    
    std::cout << "\n✓ Entraînement réussi" << std::endl;
    std::cout << "  Perte finale: " << losses.back() << std::endl;
    
    // Tester quelques prédictions
    std::cout << "\nQuelques prédictions:" << std::endl;
    std::vector<std::pair<float, float>> test_cases = {
        {0.0f, 0.0f}, {0.5f, 0.5f}, {1.0f, 1.0f}, {0.25f, 0.75f}
    };
    
    for (auto& test : test_cases) {
        Vector<float> input(2, std::vector<float>{test.first, test.second});
        Vector<float> prediction = nn.predict(input);
        float expected = (test.first + test.second) / 2.0f;
        std::cout << "  [" << test.first << ", " << test.second << "] -> " 
                  << std::fixed << std::setprecision(4) << prediction(0) 
                  << " (attendu: " << expected << ")" << std::endl;
    }
}

int main() {
    std::cout << "╔════════════════════════════════════════════════╗" << std::endl;
    std::cout << "║   Tests du Réseau de Neurones (From Scratch)  ║" << std::endl;
    std::cout << "╚════════════════════════════════════════════════╝" << std::endl;
    
    try {
        testConstruction();
        testForward();
        testLearningRateModification();
        testSaveLoad();
        testSimpleFunction();
        testTrainingXOR();
        
        std::cout << "\n╔════════════════════════════════════════════════╗" << std::endl;
        std::cout << "║        ✓ Tous les tests sont réussis !        ║" << std::endl;
        std::cout << "╚════════════════════════════════════════════════╝" << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Erreur lors des tests: " << e.what() << std::endl;
        return 1;
    }
}
