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

namespace py = pybind11;

// Global neural network instance (loaded once)
static std::unique_ptr<NeuralNetwork<float>> g_network = nullptr;
static size_t g_latentLayerIndex = 0;

/**
 * @brief Initialize the decoder by loading neural network weights
 */
void init_decoder(const std::string& model_path, size_t latent_layer_index = 3) {
    g_network = std::make_unique<NeuralNetwork<float>>(model_path);
    g_latentLayerIndex = latent_layer_index;
    
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
    
    // Create Vector from latent values
    Vector<float> latent(latent_size);
    for (size_t i = 0; i < latent_size; ++i) {
        latent(i) = latent_values[i];
    }
    
    // Decode: latent -> output
    Vector<float> output = g_network->decodeLatent(latent, g_latentLayerIndex);
    
    // Output format: [all_X, all_Y, all_Z] stacked
    size_t output_size = output.getSize();
    size_t n_points = output_size / 3;
    
    // Destandardization constants (from femur.cpp)
    // X: (val - 7.9) / 246.4  => val = output * 246.4 + 7.9
    // Y: (val - 6.4) / 61.1   => val = output * 61.1 + 6.4
    // Z: (val - 6.8) / 44.8   => val = output * 44.8 + 6.8
    constexpr float scale_x = 246.4f, offset_x = 7.9f;
    constexpr float scale_y = 61.1f, offset_y = 6.4f;
    constexpr float scale_z = 44.8f, offset_z = 6.8f;
    
    // Create numpy array (n_points, 3)
    py::array_t<float> result({n_points, size_t(3)});
    auto buf = result.mutable_unchecked<2>();
    
    // Reshape and destandardize: [all_X, all_Y, all_Z] -> (n_points, 3)
    for (size_t i = 0; i < n_points; ++i) {
        buf(i, 0) = output(i) * scale_x + offset_x;
        buf(i, 1) = output(n_points + i) * scale_y + offset_y;
        buf(i, 2) = output(2 * n_points + i) * scale_z + offset_z;
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
    
    // Standardization constants (from femur.cpp)
    constexpr float scale_x = 246.4f, offset_x = 7.9f;
    constexpr float scale_y = 61.1f, offset_y = 6.4f;
    constexpr float scale_z = 44.8f, offset_z = 6.8f;
    
    // Create input vector: [all_X, all_Y, all_Z] standardized
    Vector<float> input(n_points * 3);
    for (size_t i = 0; i < n_points; ++i) {
        input(i) = (buf(i, 0) - offset_x) / scale_x;
        input(n_points + i) = (buf(i, 1) - offset_y) / scale_y;
        input(2 * n_points + i) = (buf(i, 2) - offset_z) / scale_z;
    }
    
    // Get weights and biases (copies to avoid const issues)
    std::vector<Matrix2D<float>> weights = g_network->getWeights();
    std::vector<Vector<float>> biases = g_network->getBiases();
    
    // Forward pass up to latent layer
    Vector<float> current = input;
    for (size_t layer = 0; layer < g_latentLayerIndex; ++layer) {
        Vector<float> z = weights[layer] * current + biases[layer];
        // Apply activation (tanh)
        current = Vector<float>(z.getSize());
        for (size_t i = 0; i < z.getSize(); ++i) {
            current(i) = std::tanh(z(i));
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
          "Initialize the decoder by loading neural network weights");
    
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
}
