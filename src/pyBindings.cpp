/**
 * @file pyBindings.cpp
 * @brief Python bindings for the neural network decoder using pybind11
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "neuralNetwork.hpp"
#include "linalg.hpp"
#include <memory>
#include <stdexcept>
#include <fstream>
#include <sstream>

namespace py = pybind11;

// Global neural network instance (loaded once)
static std::unique_ptr<NeuralNetwork<float>> g_network = nullptr;
static size_t g_latentLayerIndex = 0;
static std::unique_ptr<Vector<float>> g_meanFemurCoords;
static float g_maxDifference = 36.0f; // Default, can be set in init_decoder

// Helper to load mean femur coordinates from OBJ
Vector<float> load_mean_femur(const std::string& mean_femur_path) {
    std::ifstream file(mean_femur_path);
    if (!file.is_open()) throw std::runtime_error("Could not open mean femur OBJ: " + mean_femur_path);
    std::string line;
    std::vector<float> coords;
    while (std::getline(file, line)) {
        if (line.size() > 2 && line[0] == 'v' && line[1] == ' ') {
            std::istringstream iss(line.substr(2));
            float x, y, z;
            iss >> x >> y >> z;
            coords.push_back(x);
            coords.push_back(y);
            coords.push_back(z);
        }
    }
    file.close();
    Vector<float> result(coords.size());
    for (size_t i = 0; i < coords.size(); ++i) result(i) = coords[i];
    return result;
}

/**
 * @brief Initialize the decoder by loading neural network weights
 */
void init_decoder(const std::string& model_path, size_t latent_layer_index = 3, const std::string& mean_femur_path = "../data/mean_femur.obj", float max_difference = 36.0f) {
    g_network = std::make_unique<NeuralNetwork<float>>(model_path);
    g_latentLayerIndex = latent_layer_index;
    g_meanFemurCoords = std::make_unique<Vector<float>>(load_mean_femur(mean_femur_path));
    g_maxDifference = max_difference;

    const auto& layers = g_network->getLayers();
    std::cout << "[C++] Network loaded. Architecture: ";
    for (size_t i = 0; i < layers.size(); ++i) {
        std::cout << layers[i];
        if (i < layers.size() - 1) std::cout << " -> ";
    }
    std::cout << std::endl;
    std::cout << "[C++] Latent layer index: " << latent_layer_index 
              << " (size: " << layers[latent_layer_index] << ")" << std::endl;
}

/**
 * @brief Decode a latent vector to 3D coordinates
 */
py::array_t<float> decode(const std::vector<float>& latent_values) {
    if (!g_network) {
        throw std::runtime_error("Decoder not initialized. Call init_decoder() first.");
    }
    const auto& layers = g_network->getLayers();
    size_t latent_size = layers[g_latentLayerIndex];
    if (latent_values.size() != latent_size) {
        throw std::runtime_error(
            "Latent vector size mismatch. Expected " + std::to_string(latent_size) +
            ", got " + std::to_string(latent_values.size())
        );
    }
    Vector<float> latent(latent_size);
    for (size_t i = 0; i < latent_size; ++i) latent(i) = latent_values[i];
    Vector<float> output = g_network->decodeLatent(latent, g_latentLayerIndex);
    // output is normalized: (coords - mean) / maxDifference
    // destandardize: coords = output * maxDifference + mean
    size_t output_size = output.getSize();
    size_t n_points = output_size / 3;
    py::array_t<float> result({n_points, size_t(3)});
    auto buf = result.mutable_unchecked<2>();
    // Ensure mean femur size matches expected
    if (g_meanFemurCoords->getSize() != n_points * 3) {
        throw std::runtime_error("Mean femur coordinates size mismatch. Expected " + std::to_string(n_points * 3) + ", got " + std::to_string(g_meanFemurCoords->getSize()));
    }
    for (size_t i = 0; i < n_points; ++i) {
        buf(i, 0) = output(i) * g_maxDifference + (*g_meanFemurCoords)(i*3);
        buf(i, 1) = output(n_points + i) * g_maxDifference + (*g_meanFemurCoords)(i*3+1);
        buf(i, 2) = output(2 * n_points + i) * g_maxDifference + (*g_meanFemurCoords)(i*3+2);
    }
    return result;
}

size_t get_latent_size() {
    if (!g_network) {
        throw std::runtime_error("Decoder not initialized. Call init_decoder() first.");
    }
    return g_network->getLayers()[g_latentLayerIndex];
}

size_t get_num_points() {
    if (!g_network) {
        throw std::runtime_error("Decoder not initialized. Call init_decoder() first.");
    }
    return g_network->getLayers().back() / 3;
}

/**
 * @brief Get the activation function name used in the network
 */
std::string get_activation_function() {
    if (!g_network) {
        throw std::runtime_error("Decoder not initialized. Call init_decoder() first.");
    }
    return g_network->getActivation();
}

/**
 * @brief Encode 3D coordinates to latent vector
 * Takes vertices in real coordinates (not standardized) and returns latent vector
 */
std::vector<float> encode(py::array_t<float> vertices) {
    if (!g_network) {
        throw std::runtime_error("Decoder not initialized. Call init_decoder() first.");
    }
    auto buf = vertices.unchecked<2>();
    size_t n_points = buf.shape(0);
    size_t expected_points = g_network->getLayers()[0] / 3;
    if (n_points != expected_points) {
        throw std::runtime_error(
            "Vertex count mismatch. Expected " + std::to_string(expected_points) +
            ", got " + std::to_string(n_points)
        );
    }
    // Standardize: (coords - mean) / maxDifference
    Vector<float> input(n_points * 3);
    for (size_t i = 0; i < n_points; ++i) {
        input(i) = (buf(i, 0) - (*g_meanFemurCoords)(i*3)) / g_maxDifference;
        input(n_points + i) = (buf(i, 1) - (*g_meanFemurCoords)(i*3+1)) / g_maxDifference;
        input(2 * n_points + i) = (buf(i, 2) - (*g_meanFemurCoords)(i*3+2)) / g_maxDifference;
    }
    // Get weights and biases (copies to avoid const issues)
    std::vector<Matrix2D<float>> weights = g_network->getWeights();
    std::vector<Vector<float>> biases = g_network->getBiases();
    // Forward pass up to latent layer
    Vector<float> current = input;
    std::string activation = g_network->getActivation();
    for (size_t layer = 0; layer < g_latentLayerIndex; ++layer) {
        Vector<float> z = weights[layer] * current + biases[layer];
        // Apply activation based on network's activation function
        current = Vector<float>(z.getSize());
        for (size_t i = 0; i < z.getSize(); ++i) {
            if (activation == "sigmoid") {
                current(i) = 1.0f / (1.0f + std::exp(-z(i)));
            } else if (activation == "tanh") {
                current(i) = std::tanh(z(i));
            } else if (activation == "ReLU") {
                current(i) = std::max(0.0f, z(i));
            } else if (activation == "LeakyReLU") {
                current(i) = z(i) > 0 ? z(i) : 0.01f * z(i);
            } else {
                current(i) = std::tanh(z(i));  // fallback to tanh
            }
        }
    }
    // current now contains the latent representation
    std::vector<float> result(current.getSize());
    for (size_t i = 0; i < current.getSize(); ++i) {
        result[i] = current(i);
    }
    return result;
}

// Python module definition
PYBIND11_MODULE(femur_rdn, m) {
    m.doc() = "Femur RDN decoder - Python bindings for latent space exploration";
    m.def("init_decoder", &init_decoder,
          py::arg("model_path"),
          py::arg("latent_layer_index") = 3,
          py::arg("mean_femur_path") = "../data/mean_femur.obj",
          py::arg("max_difference") = 36.0f,
          "Initialize the decoder by loading neural network weights and normalization info");
    m.def("decode", &decode,
          py::arg("latent_values"),
          "Decode a latent vector to 3D coordinates");
    m.def("get_latent_size", &get_latent_size,
          "Get the number of neurons in the latent layer");
    m.def("get_num_points", &get_num_points,
          "Get the number of 3D points the decoder produces");
    m.def("encode", &encode,
          py::arg("vertices"),
          "Encode 3D vertices (N,3) to latent vector");
    m.def("get_activation_function", &get_activation_function,
          "Get the activation function name used in the network (sigmoid, tanh, ReLU, LeakyReLU)");
}
