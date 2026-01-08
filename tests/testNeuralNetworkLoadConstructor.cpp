#include <iostream>
#include <cassert>
#include <cstdio>    // std::remove
#include <iomanip>
#include "neuralNetwork.hpp"
#include "linalg.hpp"

template <typename T>
bool isApprox(T a, T b, T epsilon = static_cast<T>(1e-5)) {
    return std::abs(a - b) < epsilon;
}

int main() {
    std::cout << "=== Test: Constructeur de chargement et entraînement ultérieur ===\n";

    // 1) Créer et entraîner un réseau
    std::vector<size_t> layers = {2, 3, 1};
    NeuralNetwork<float> nn(layers, 0.3f);

    std::vector<Vector<float>> inputs;
    std::vector<Vector<float>> targets;

    // Jeu XOR
    inputs.push_back(Vector<float>(2, std::vector<float>{0.0f, 0.0f})); targets.push_back(Vector<float>(1, std::vector<float>{0.0f}));
    inputs.push_back(Vector<float>(2, std::vector<float>{0.0f, 1.0f})); targets.push_back(Vector<float>(1, std::vector<float>{1.0f}));
    inputs.push_back(Vector<float>(2, std::vector<float>{1.0f, 0.0f})); targets.push_back(Vector<float>(1, std::vector<float>{1.0f}));
    inputs.push_back(Vector<float>(2, std::vector<float>{1.0f, 1.0f})); targets.push_back(Vector<float>(1, std::vector<float>{0.0f}));

    std::vector<float> losses_before = nn.train(inputs, targets, 500, false);
    std::cout << "Perte après pré-entraînement: " << losses_before.back() << "\n";

    // 2) Sauvegarder le réseau
    std::string filename = "constructor_network.txt";
    bool saved = nn.save(filename);
    assert(saved);

    // 3) Recharger via le constructeur qui prend le nom de fichier
    NeuralNetwork<float> nn_loaded(filename);

    // 4) Vérifier que les prédictions sont identiques avant entraînement supplémentaire
    for (size_t i = 0; i < inputs.size(); ++i) {
        float p1 = nn.predict(inputs[i])(0);
        float p2 = nn_loaded.predict(inputs[i])(0);
        assert(isApprox(p1, p2, 1e-4f));
    }
    std::cout << "✓ Prédictions identiques après chargement via constructeur.\n";

    // 5) Continuer l'entraînement sur une époque
    std::vector<float> losses_after = nn_loaded.train(inputs, targets, 1, true);
    std::cout << "Perte après 1 époque supplémentaire: " << losses_after.back() << "\n";

    // Nettoyage
    std::remove(filename.c_str());
    std::cout << "✓ Test terminé.\n";

    return 0;
}