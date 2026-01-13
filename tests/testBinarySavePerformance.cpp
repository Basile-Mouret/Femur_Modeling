#include <iostream>
#include <chrono>
#include <filesystem>
#include <vector>
#include <cmath>
#include "neuralNetwork.hpp"

// Utility to measure time
template<typename Func>
double measure_time(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;
    return duration.count();
}

int main() {
    std::cout << "=======================================" << std::endl;
    std::cout << "Testing Binary Save Performance vs Text" << std::endl;
    std::cout << "=======================================" << std::endl;

    // Create a reasonably large network to make the difference noticeable
    // Layers: Input(1000) -> 500 -> 500 -> 500 -> Output(100)
    // Weights: 1000*500 + 500*500 + 500*500 + 500*100 = 500k + 250k + 250k + 50k = 1.05 million parameters
    std::vector<size_t> layers = {1000, 500, 500, 500, 100};
    
    std::cout << "Initializing Neural Network with layers: ";
    for (size_t s : layers) std::cout << s << " ";
    std::cout << std::endl;

    NeuralNetwork<float> nn(layers, "sigmoid", "meanSquaredError", 0.01f);
    
    // We don't need to train it, random weights are fine for testing IO performance

    std::string textFilename = "test_network_perf.nn";
    std::string binaryFilename = "test_network_perf.bin";

    std::cout << "\n1. Measuring Text Save Performance..." << std::endl;
    double textDuration = measure_time([&]() {
        nn.save(textFilename);
    });
    std::cout << "   Time: " << textDuration << " ms" << std::endl;

    std::cout << "\n2. Measuring Binary Save Performance..." << std::endl;
    double binaryDuration = measure_time([&]() {
        nn.saveBinary(binaryFilename);
    });
    std::cout << "   Time: " << binaryDuration << " ms" << std::endl;

    // Compare
    std::cout << "\n---------------------------------------" << std::endl;
    std::cout << "Results:" << std::endl;
    double speedup = textDuration / binaryDuration;
    std::cout << "Speedup: " << speedup << "x faster" << std::endl;

    if (binaryDuration < textDuration) {
        std::cout << "SUCCESS: Binary save is faster!" << std::endl;
    } else {
        std::cerr << "WARNING: Binary save was NOT faster (might be file caching or small network)." << std::endl;
    }

    // Check file sizes
    try {
        auto textSize = std::filesystem::file_size(textFilename);
        auto binarySize = std::filesystem::file_size(binaryFilename);
        
        std::cout << "\nFile Sizes:" << std::endl;
        std::cout << "Text File:   " << textSize / 1024.0 / 1024.0 << " MB" << std::endl;
        std::cout << "Binary File: " << binarySize / 1024.0 / 1024.0 << " MB" << std::endl;
        std::cout << "Size Reduction: " << (double)textSize / binarySize << "x smaller" << std::endl;
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error checking file sizes: " << e.what() << std::endl;
    }

    // Verification - Load Binary to ensure it works
    std::cout << "\nVerifying Binary Load..." << std::endl;
    try {
        NeuralNetwork<float> nnLoaded(binaryFilename);
        std::cout << "SUCCESS: Binary file loaded correctly." << std::endl;
    } catch (...) {
        std::cerr << "FAILURE: Failed to load binary file." << std::endl;
    }

    // Cleanup
    std::cout << "\nCleaning up..." << std::endl;
    try {
        std::filesystem::remove(textFilename);
        std::filesystem::remove(binaryFilename);
    } catch (const std::filesystem::filesystem_error& e) {
        std::cerr << "Error removing files: " << e.what() << std::endl;
    }

    return 0;
}
