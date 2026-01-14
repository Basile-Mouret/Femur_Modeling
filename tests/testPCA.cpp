/**
 * @file testPCA.cpp
 * @brief Unit tests for PCA implementation
 * @details Tests PCA fitting, transformation, reconstruction, and I/O
 */

#include <iostream>
#include <cassert>
#include <cmath>
#include <iomanip>
#include "../include/pca.hpp"
#include "../include/dataset.hpp"
#include "../include/femur.hpp"

// Test tolerance for floating point comparisons
constexpr double TOLERANCE = 1e-6;

/**
 * @brief Helper function to check if two values are approximately equal
 */
bool approxEqual(double a, double b, double tol = TOLERANCE) {
    return std::abs(a - b) < tol;
}

/**
 * @brief Test PCA with simple synthetic data
 */
void testPCASyntheticData() {
    std::cout << "\n=== Test: PCA with Synthetic Data ===" << std::endl;
    
    // Create simple 2D data with clear principal directions
    // Data: 4 points forming an elongated cloud along x-axis
    Matrix2D<double> data(2, 4);
    
    // Points: (-2, 0.1), (-1, -0.1), (1, 0.1), (2, -0.1)
    // First PC should be along x-axis, second along y-axis
    data.setCoeff(0, 0, -2.0); data.setCoeff(1, 0,  0.1);  // Point 1
    data.setCoeff(0, 1, -1.0); data.setCoeff(1, 1, -0.1);  // Point 2
    data.setCoeff(0, 2,  1.0); data.setCoeff(1, 2,  0.1);  // Point 3
    data.setCoeff(0, 3,  2.0); data.setCoeff(1, 3, -0.1);  // Point 4
    
    // Fit PCA
    PCA<double> pca;
    pca.fit(data);
    
    // Check basic properties
    assert(pca.isFitted() && "PCA should be fitted");
    assert(pca.getNumSamples() == 4 && "Should have 4 samples");
    assert(pca.getNumDimensions() == 2 && "Should have 2 dimensions");
    assert(pca.getNumComponents() >= 1 && "Should have at least 1 component");
    
    // Check mean (should be approximately (0, 0))
    Vector<double> mean = pca.getMean();
    assert(approxEqual(mean(0), 0.0, 0.01) && "Mean x should be ~0");
    assert(approxEqual(mean(1), 0.0, 0.01) && "Mean y should be ~0");
    
    // Check that first PC captures most variance
    Vector<double> varRatios = pca.explainedVarianceRatio();
    assert(varRatios(0) > 0.9 && "First PC should explain >90% variance");
    
    std::cout << "Mean: (" << mean(0) << ", " << mean(1) << ")" << std::endl;
    std::cout << "Variance ratios: " << varRatios(0) * 100 << "%, " 
              << (pca.getNumComponents() > 1 ? varRatios(1) * 100 : 0) << "%" << std::endl;
    
    std::cout << "✓ Synthetic data test passed!" << std::endl;
}

/**
 * @brief Test transform and inverse transform are consistent
 */
void testTransformConsistency() {
    std::cout << "\n=== Test: Transform Consistency ===" << std::endl;
    
    // Create random data
    const size_t D = 10;  // dimensions
    const size_t N = 5;   // samples
    
    Matrix2D<double> data(D, N);
    std::mt19937 rng(42);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < N; ++j) {
            data.setCoeff(i, j, dist(rng));
        }
    }
    
    // Fit PCA
    PCA<double> pca;
    pca.fit(data);
    
    // Test each data point
    for (size_t j = 0; j < N; ++j) {
        Vector<double> original(D);
        for (size_t i = 0; i < D; ++i) {
            original(i) = data(i, j);
        }
        
        // Transform to PCA space and back
        Vector<double> coeffs = pca.transform(original);
        Vector<double> reconstructed = pca.inverseTransform(coeffs);
        
        // Check reconstruction error (should be near zero with all components)
        double error = 0.0;
        for (size_t i = 0; i < D; ++i) {
            double diff = original(i) - reconstructed(i);
            error += diff * diff;
        }
        error = std::sqrt(error / D);
        
        assert(error < 1e-10 && "Reconstruction error should be near zero");
    }
    
    std::cout << "✓ Transform consistency test passed!" << std::endl;
}

/**
 * @brief Test reconstruction with varying number of components
 */
void testPartialReconstruction() {
    std::cout << "\n=== Test: Partial Reconstruction ===" << std::endl;
    
    // Create data with known variance structure
    const size_t D = 20;
    const size_t N = 10;
    
    Matrix2D<double> data(D, N);
    std::mt19937 rng(123);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    // Add decreasing variance in each dimension
    for (size_t i = 0; i < D; ++i) {
        double scale = 1.0 / (i + 1);  // Decreasing variance
        for (size_t j = 0; j < N; ++j) {
            data.setCoeff(i, j, dist(rng) * scale);
        }
    }
    
    PCA<double> pca;
    pca.fit(data);
    
    // Test reconstruction error decreases with more components
    Vector<double> testShape(D);
    for (size_t i = 0; i < D; ++i) {
        testShape(i) = data(i, 0);  // First sample
    }
    
    double prevError = std::numeric_limits<double>::max();
    std::cout << "Reconstruction errors by component count:" << std::endl;
    
    for (size_t k = 1; k <= pca.getNumComponents(); k += 2) {
        double error = pca.reconstructionError(testShape, k);
        std::cout << "  K=" << k << ": MSE = " << std::scientific << error << std::endl;
        assert(error <= prevError + 1e-10 && "Error should decrease with more components");
        prevError = error;
    }
    
    std::cout << "✓ Partial reconstruction test passed!" << std::endl;
}

/**
 * @brief Test shape generation methods
 */
void testShapeGeneration() {
    std::cout << "\n=== Test: Shape Generation ===" << std::endl;
    
    const size_t D = 15;
    const size_t N = 6;
    
    Matrix2D<double> data(D, N);
    std::mt19937 rng(456);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < N; ++j) {
            data.setCoeff(i, j, dist(rng));
        }
    }
    
    PCA<double> pca;
    pca.fit(data);
    
    // Test generateAlongMode
    Vector<double> meanShape = pca.getMean();
    
    for (size_t mode = 0; mode < std::min(pca.getNumComponents(), size_t(3)); ++mode) {
        Vector<double> minusShape = pca.generateAlongMode(mode, -2.0);
        Vector<double> plusShape = pca.generateAlongMode(mode, 2.0);
        
        // Check that generated shapes differ from mean along the mode
        double diffMinus = 0, diffPlus = 0;
        for (size_t i = 0; i < D; ++i) {
            diffMinus += (minusShape(i) - meanShape(i)) * (minusShape(i) - meanShape(i));
            diffPlus += (plusShape(i) - meanShape(i)) * (plusShape(i) - meanShape(i));
        }
        
        assert(diffMinus > 1e-10 && "Minus shape should differ from mean");
        assert(diffPlus > 1e-10 && "Plus shape should differ from mean");
        
        std::cout << "  Mode " << mode << ": |minus - mean| = " << std::sqrt(diffMinus) 
                  << ", |plus - mean| = " << std::sqrt(diffPlus) << std::endl;
    }
    
    // Test generateShape with weights
    Vector<double> weights(pca.getNumComponents(), 0.0);
    weights(0) = 1.0;  // Activate first mode
    Vector<double> generated = pca.generateShape(weights);
    assert(generated.getSize() == D && "Generated shape should have correct dimensions");
    
    // Test sampleShapes
    Matrix2D<double> samples = pca.sampleShapes(5, 3);
    assert(samples.getSizeRows() == D && "Samples should have correct dimensions");
    assert(samples.getSizeCols() == 5 && "Should generate 5 samples");
    
    std::cout << "✓ Shape generation test passed!" << std::endl;
}

/**
 * @brief Test save and load functionality
 */
void testSaveLoad() {
    std::cout << "\n=== Test: Save/Load ===" << std::endl;
    
    const size_t D = 10;
    const size_t N = 5;
    
    Matrix2D<double> data(D, N);
    std::mt19937 rng(789);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < N; ++j) {
            data.setCoeff(i, j, dist(rng));
        }
    }
    
    // Fit and save
    PCA<double> pca1;
    pca1.fit(data);
    pca1.save("/tmp/test_pca_model.bin");
    
    // Load into new object
    PCA<double> pca2("/tmp/test_pca_model.bin");
    
    // Verify loaded model matches original
    assert(pca2.isFitted() && "Loaded model should be fitted");
    assert(pca2.getNumDimensions() == pca1.getNumDimensions() && "Dimensions should match");
    assert(pca2.getNumComponents() == pca1.getNumComponents() && "Components should match");
    assert(pca2.getNumSamples() == pca1.getNumSamples() && "Samples should match");
    
    // Compare means
    Vector<double> mean1 = pca1.getMean();
    Vector<double> mean2 = pca2.getMean();
    double meanDiff = 0;
    for (size_t i = 0; i < D; ++i) {
        meanDiff += (mean1(i) - mean2(i)) * (mean1(i) - mean2(i));
    }
    assert(meanDiff < 1e-20 && "Means should be identical");
    
    // Compare transformations
    Vector<double> testShape(D);
    for (size_t i = 0; i < D; ++i) {
        testShape(i) = dist(rng);
    }
    
    Vector<double> coeffs1 = pca1.transform(testShape);
    Vector<double> coeffs2 = pca2.transform(testShape);
    
    double coeffDiff = 0;
    for (size_t i = 0; i < coeffs1.getSize(); ++i) {
        coeffDiff += (coeffs1(i) - coeffs2(i)) * (coeffs1(i) - coeffs2(i));
    }
    assert(coeffDiff < 1e-20 && "Coefficients should be identical");
    
    std::cout << "✓ Save/Load test passed!" << std::endl;
}

/**
 * @brief Test variance analysis methods
 */
void testVarianceAnalysis() {
    std::cout << "\n=== Test: Variance Analysis ===" << std::endl;
    
    const size_t D = 8;
    const size_t N = 20;
    
    Matrix2D<double> data(D, N);
    std::mt19937 rng(101112);
    std::normal_distribution<double> dist(0.0, 1.0);
    
    for (size_t i = 0; i < D; ++i) {
        for (size_t j = 0; j < N; ++j) {
            data.setCoeff(i, j, dist(rng));
        }
    }
    
    PCA<double> pca;
    pca.fit(data);
    
    // Check explained variance ratios sum to 1
    Vector<double> ratios = pca.explainedVarianceRatio();
    double sum = 0;
    for (size_t i = 0; i < ratios.getSize(); ++i) {
        sum += ratios(i);
        assert(ratios(i) >= 0 && "Variance ratios should be non-negative");
    }
    assert(approxEqual(sum, 1.0, 0.01) && "Variance ratios should sum to ~1");
    
    // Check cumulative variance is monotonically increasing
    Vector<double> cumulative = pca.cumulativeVarianceRatio();
    for (size_t i = 1; i < cumulative.getSize(); ++i) {
        assert(cumulative(i) >= cumulative(i-1) && "Cumulative variance should increase");
    }
    assert(approxEqual(cumulative(cumulative.getSize()-1), 1.0, 0.01) && 
           "Final cumulative should be ~1");
    
    // Test componentsForVariance
    size_t k90 = pca.componentsForVariance(0.90);
    size_t k95 = pca.componentsForVariance(0.95);
    size_t k99 = pca.componentsForVariance(0.99);
    
    assert(k90 <= k95 && k95 <= k99 && "Higher threshold needs more components");
    assert(k99 <= pca.getNumComponents() && "Should not exceed total components");
    
    std::cout << "  Variance ratios sum: " << sum << std::endl;
    std::cout << "  Components for 90%: " << k90 << std::endl;
    std::cout << "  Components for 95%: " << k95 << std::endl;
    std::cout << "  Components for 99%: " << k99 << std::endl;
    
    std::cout << "✓ Variance analysis test passed!" << std::endl;
}

/**
 * @brief Test with FemurDataset (integration test)
 */
void testWithFemurDataset() {
    std::cout << "\n=== Test: Integration with FemurDataset ===" << std::endl;
    
    // Try to load real data - use absolute path for robustness
    FemurDataset<double> dataset;
    size_t loaded = dataset.loadFromDirectory("/home/tag/Desktop/Femur_Modeling/data/training/", true, 1);
    
    if (loaded < 2) {
        std::cout << "⚠ Skipping: Not enough training data available" << std::endl;
        std::cout << "  (Need at least 2 shapes, found " << loaded << ")" << std::endl;
        return;
    }
    
    dataset.printInfo();
    
    // Convert to matrix and fit PCA
    Matrix2D<double> dataMatrix = dataset.toMatrix();
    std::cout << "Data matrix size: " << dataMatrix.getSizeRows() << " x " 
              << dataMatrix.getSizeCols() << std::endl;
    
    PCA<double> pca;
    pca.fit(dataMatrix);
    pca.printSummary();
    
    // Test reconstruction on first shape
    Vector<double> original = dataset.getShapeVector(0);
    Vector<double> reconstructed = pca.reconstruct(original);
    
    double mse = 0;
    for (size_t i = 0; i < original.getSize(); ++i) {
        double diff = original(i) - reconstructed(i);
        mse += diff * diff;
    }
    mse /= original.getSize();
    
    std::cout << "Reconstruction MSE (all components): " << std::scientific << mse << std::endl;
    
    // Test partial reconstruction
    std::cout << "Reconstruction MSE by components:" << std::endl;
    for (int k : {1, 3, 5, 10, 15, 20}) {
        if (static_cast<size_t>(k) > pca.getNumComponents()) break;
        double error = pca.reconstructionError(original, k);
        std::cout << "  K=" << k << ": " << std::scientific << error << std::endl;
    }
    
    // Save model
    pca.save("/home/tag/Desktop/Femur_Modeling/bin/pca_femur_model.bin");
    std::cout << "✓ Model saved to bin/pca_femur_model.bin" << std::endl;
    
    std::cout << "✓ FemurDataset integration test passed!" << std::endl;
}

/**
 * @brief Main test runner
 */
int main() {
    std::cout << "\n"
              << "╔══════════════════════════════════════════════════════════════╗\n"
              << "║              PCA Implementation Tests                        ║\n"
              << "╚══════════════════════════════════════════════════════════════╝\n"
              << std::endl;
    
    try {
        testPCASyntheticData();
        testTransformConsistency();
        testPartialReconstruction();
        testShapeGeneration();
        testSaveLoad();
        testVarianceAnalysis();
        testWithFemurDataset();
        
        std::cout << "\n"
                  << "╔══════════════════════════════════════════════════════════════╗\n"
                  << "║              All tests passed! ✓                             ║\n"
                  << "╚══════════════════════════════════════════════════════════════╝\n"
                  << std::endl;
        
        return 0;
        
    } catch (const std::exception& e) {
        std::cerr << "\n✗ Test failed with exception: " << e.what() << std::endl;
        return 1;
    }
}
