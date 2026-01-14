/**
 * @file pca.hpp
 * @brief Principal Component Analysis for Statistical Shape Modeling
 * @author Femur Modeling Project
 * @date 2024
 * 
 * This file contains a template-based PCA implementation optimized for
 * statistical shape analysis. Uses SVD for efficient computation when
 * the number of dimensions D >> number of samples N.
 */

#ifndef PCA_HPP
#define PCA_HPP

#include <vector>
#include <cmath>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <string>

// Eigen for SVD computation
#include <Eigen/Dense>
#include <Eigen/SVD>

#include "linalg.hpp"

/**
 * @class PCA
 * @brief Template class for Principal Component Analysis
 * 
 * This class implements PCA using Singular Value Decomposition (SVD),
 * which is particularly efficient when D >> N (high-dimensional data
 * with few samples, as is typical in shape analysis).
 * 
 * @tparam T Numeric type (float or double)
 * 
 * Mathematical Background:
 * Given a data matrix X (D x N) with N samples in D dimensions:
 * 1. Center the data: X' = X - mean
 * 2. Compute SVD: X' = U * S * V^T
 * 3. Principal components are columns of U
 * 4. Variances are S^2 / (N-1)
 * 
 * For shape modeling, a shape s can be represented as:
 *   s = mean + sum_k(alpha_k * sqrt(lambda_k) * v_k)
 * where v_k are principal components and lambda_k are variances.
 */
template <typename T>
class PCA {
private:
    Vector<T> m_mean;              ///< Mean shape (D x 1)
    Matrix2D<T> m_components;      ///< Principal components (D x K), columns are eigenvectors
    Vector<T> m_variances;         ///< Eigenvalues / variances for each component
    Vector<T> m_singularValues;    ///< Singular values from SVD
    
    size_t m_numSamples;           ///< Number of training samples (N)
    size_t m_numDimensions;        ///< Dimensionality of data (D)
    size_t m_numComponents;        ///< Number of components retained (K)
    T m_totalVariance;             ///< Total variance in the data
    bool m_fitted;                 ///< Whether the model has been fitted
    
    mutable std::mt19937 m_rng;    ///< Random number generator for sampling

public:
    /**
     * @brief Default constructor
     * Creates an unfitted PCA model.
     */
    PCA() : m_mean(1), m_components(1, 1), m_variances(1), m_singularValues(1),
            m_numSamples(0), m_numDimensions(0), m_numComponents(0),
            m_totalVariance(0), m_fitted(false), m_rng(std::random_device{}()) {}
    
    /**
     * @brief Constructor that loads from file
     * @param filename Path to saved PCA model
     */
    explicit PCA(const std::string& filename) 
        : m_mean(1), m_components(1, 1), m_variances(1), m_singularValues(1),
          m_numSamples(0), m_numDimensions(0), m_numComponents(0),
          m_totalVariance(0), m_fitted(false), m_rng(std::random_device{}()) {
        load(filename);
    }
    
    /**
     * @brief Fits the PCA model to data
     * 
     * Computes the mean and principal components from the data matrix
     * using SVD decomposition.
     * 
     * @param data Data matrix (D x N) where each column is a sample
     * @param maxComponents Maximum number of components to retain (-1 = all)
     */
    void fit(const Matrix2D<T>& data, int maxComponents = -1) {
        if (data.getSizeCols() < 2) {
            throw std::invalid_argument("PCA requires at least 2 samples");
        }
        
        m_numDimensions = data.getSizeRows();
        m_numSamples = data.getSizeCols();
        
        std::cout << "Fitting PCA: D=" << m_numDimensions 
                  << ", N=" << m_numSamples << std::endl;
        
        // Step 1: Compute mean
        m_mean = Vector<T>(m_numDimensions);
        for (size_t i = 0; i < m_numDimensions; ++i) {
            T sum = 0;
            for (size_t j = 0; j < m_numSamples; ++j) {
                sum += data(i, j);
            }
            m_mean(i) = sum / static_cast<T>(m_numSamples);
        }
        
        // Step 2: Center the data and convert to Eigen matrix
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> centered(m_numDimensions, m_numSamples);
        for (size_t i = 0; i < m_numDimensions; ++i) {
            for (size_t j = 0; j < m_numSamples; ++j) {
                centered(i, j) = data(i, j) - m_mean(i);
            }
        }
        
        // Step 3: Compute SVD
        // For D >> N, we use thin SVD which is more efficient
        std::cout << "Computing SVD..." << std::endl;
        Eigen::JacobiSVD<Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>> svd(
            centered, Eigen::ComputeThinU | Eigen::ComputeThinV);
        
        auto singularValues = svd.singularValues();
        auto U = svd.matrixU();
        
        // Step 4: Determine number of components to keep
        size_t maxK = std::min(m_numDimensions, m_numSamples - 1);
        if (maxComponents > 0) {
            m_numComponents = std::min(static_cast<size_t>(maxComponents), maxK);
        } else {
            m_numComponents = maxK;
        }
        
        std::cout << "Retaining " << m_numComponents << " components" << std::endl;
        
        // Step 5: Store singular values
        m_singularValues = Vector<T>(m_numComponents);
        for (size_t i = 0; i < m_numComponents; ++i) {
            m_singularValues(i) = singularValues(i);
        }
        
        // Step 6: Compute variances (eigenvalues)
        // Variance = singular_value^2 / (N - 1)
        m_variances = Vector<T>(m_numComponents);
        m_totalVariance = 0;
        for (size_t i = 0; i < m_numComponents; ++i) {
            m_variances(i) = (singularValues(i) * singularValues(i)) / 
                             static_cast<T>(m_numSamples - 1);
            m_totalVariance += m_variances(i);
        }
        
        // Step 7: Store principal components (columns of U)
        m_components = Matrix2D<T>(m_numDimensions, m_numComponents);
        for (size_t i = 0; i < m_numDimensions; ++i) {
            for (size_t j = 0; j < m_numComponents; ++j) {
                m_components(i, j) = U(i, j);
            }
        }
        
        m_fitted = true;
        std::cout << "PCA fitting complete. Total variance: " << m_totalVariance << std::endl;
    }
    
    /**
     * @brief Projects a shape onto the principal component space
     * 
     * @param shape Input shape vector (D x 1)
     * @param numComponents Number of components to use (-1 = all)
     * @return Vector of PCA coefficients
     */
    Vector<T> transform(const Vector<T>& shape, int numComponents = -1) const {
        if (!m_fitted) {
            throw std::runtime_error("PCA model not fitted");
        }
        if (shape.getSize() != m_numDimensions) {
            throw std::invalid_argument("Shape dimension mismatch");
        }
        
        size_t k = (numComponents > 0) ? 
                   std::min(static_cast<size_t>(numComponents), m_numComponents) : 
                   m_numComponents;
        
        // Center the shape
        Vector<T> centered(m_numDimensions);
        for (size_t i = 0; i < m_numDimensions; ++i) {
            centered(i) = shape(i) - m_mean(i);
        }
        
        // Project onto components: coefficients = V^T * (shape - mean)
        Vector<T> coefficients(k);
        for (size_t j = 0; j < k; ++j) {
            T dot = 0;
            for (size_t i = 0; i < m_numDimensions; ++i) {
                dot += m_components(i, j) * centered(i);
            }
            coefficients(j) = dot;
        }
        
        return coefficients;
    }
    
    /**
     * @brief Reconstructs a shape from PCA coefficients
     * 
     * @param coefficients PCA coefficients
     * @return Reconstructed shape vector (D x 1)
     */
    Vector<T> inverseTransform(const Vector<T>& coefficients) const {
        if (!m_fitted) {
            throw std::runtime_error("PCA model not fitted");
        }
        
        size_t k = std::min(coefficients.getSize(), m_numComponents);
        
        // Reconstruct: shape = mean + V * coefficients
        Vector<T> shape(m_numDimensions);
        for (size_t i = 0; i < m_numDimensions; ++i) {
            shape(i) = m_mean(i);
            for (size_t j = 0; j < k; ++j) {
                shape(i) = shape(i) + m_components(i, j) * coefficients(j);
            }
        }
        
        return shape;
    }
    
    /**
     * @brief Projects and reconstructs a shape
     * 
     * @param shape Input shape vector
     * @param numComponents Number of components for reconstruction
     * @return Reconstructed shape
     */
    Vector<T> reconstruct(const Vector<T>& shape, int numComponents = -1) const {
        auto coeffs = transform(shape, numComponents);
        return inverseTransform(coeffs);
    }
    
    /**
     * @brief Computes reconstruction error
     * 
     * @param shape Original shape
     * @param numComponents Number of components used
     * @return Mean squared error
     */
    T reconstructionError(const Vector<T>& shape, int numComponents = -1) const {
        auto reconstructed = reconstruct(shape, numComponents);
        T error = 0;
        for (size_t i = 0; i < m_numDimensions; ++i) {
            T diff = shape(i) - reconstructed(i);
            error += diff * diff;
        }
        return error / static_cast<T>(m_numDimensions);
    }
    
    /**
     * @brief Gets explained variance ratio for each component
     * @return Vector of explained variance ratios
     */
    Vector<T> explainedVarianceRatio() const {
        if (!m_fitted) {
            throw std::runtime_error("PCA model not fitted");
        }
        
        Vector<T> ratios(m_numComponents);
        for (size_t i = 0; i < m_numComponents; ++i) {
            ratios(i) = m_variances(i) / m_totalVariance;
        }
        return ratios;
    }
    
    /**
     * @brief Gets cumulative explained variance ratio
     * @return Vector of cumulative ratios
     */
    Vector<T> cumulativeVarianceRatio() const {
        auto ratios = explainedVarianceRatio();
        Vector<T> cumulative(m_numComponents);
        T sum = 0;
        for (size_t i = 0; i < m_numComponents; ++i) {
            sum += ratios(i);
            cumulative(i) = sum;
        }
        return cumulative;
    }
    
    /**
     * @brief Finds number of components for given variance threshold
     * @param varianceThreshold Proportion of variance (0.0 to 1.0)
     * @return Number of components needed
     */
    size_t componentsForVariance(T varianceThreshold) const {
        auto cumulative = cumulativeVarianceRatio();
        for (size_t i = 0; i < m_numComponents; ++i) {
            if (cumulative(i) >= varianceThreshold) {
                return i + 1;
            }
        }
        return m_numComponents;
    }
    
    /**
     * @brief Generates a shape with specified mode weights
     * 
     * shape = mean + sum_k(weights[k] * sqrt(variance[k]) * component[k])
     * 
     * @param weights Vector of mode weights (typically [-3, 3])
     * @return Generated shape
     */
    Vector<T> generateShape(const Vector<T>& weights) const {
        if (!m_fitted) {
            throw std::runtime_error("PCA model not fitted");
        }
        
        size_t k = std::min(weights.getSize(), m_numComponents);
        
        Vector<T> shape(m_numDimensions);
        for (size_t i = 0; i < m_numDimensions; ++i) {
            shape(i) = m_mean(i);
            for (size_t j = 0; j < k; ++j) {
                // Scale by standard deviation (sqrt of variance)
                shape(i) = shape(i) + weights(j) * std::sqrt(m_variances(j)) * m_components(i, j);
            }
        }
        
        return shape;
    }
    
    /**
     * @brief Generates a shape along a single mode
     * 
     * @param mode Index of principal component (0-based)
     * @param sigma Number of standard deviations
     * @return Generated shape
     */
    Vector<T> generateAlongMode(size_t mode, T sigma) const {
        if (!m_fitted) {
            throw std::runtime_error("PCA model not fitted");
        }
        if (mode >= m_numComponents) {
            throw std::out_of_range("Mode index out of range");
        }
        
        Vector<T> weights(m_numComponents, static_cast<T>(0));
        weights(mode) = sigma;
        
        return generateShape(weights);
    }
    
    /**
     * @brief Samples random shapes from the statistical model
     * 
     * @param numSamples Number of shapes to generate
     * @param numComponents Number of modes to use (-1 = all)
     * @return Matrix of generated shapes (D x numSamples)
     */
    Matrix2D<T> sampleShapes(size_t numSamples, int numComponents = -1) const {
        if (!m_fitted) {
            throw std::runtime_error("PCA model not fitted");
        }
        
        size_t k = (numComponents > 0) ? 
                   std::min(static_cast<size_t>(numComponents), m_numComponents) : 
                   m_numComponents;
        
        std::normal_distribution<T> dist(0, 1);
        Matrix2D<T> samples(m_numDimensions, numSamples);
        
        for (size_t s = 0; s < numSamples; ++s) {
            // Generate random weights
            Vector<T> weights(k);
            for (size_t i = 0; i < k; ++i) {
                weights(i) = dist(m_rng);
            }
            
            // Generate shape
            auto shape = generateShape(weights);
            for (size_t i = 0; i < m_numDimensions; ++i) {
                samples(i, s) = shape(i);
            }
        }
        
        return samples;
    }
    
    // Getters
    Vector<T> getMean() const { return m_mean; }
    Matrix2D<T> getComponents() const { return m_components; }
    
    Vector<T> getComponent(size_t index) const {
        if (index >= m_numComponents) {
            throw std::out_of_range("Component index out of range");
        }
        Vector<T> component(m_numDimensions);
        for (size_t i = 0; i < m_numDimensions; ++i) {
            component(i) = m_components(i, index);
        }
        return component;
    }
    
    Vector<T> getVariances() const { return m_variances; }
    
    Vector<T> getStdDevs() const {
        Vector<T> stddevs(m_numComponents);
        for (size_t i = 0; i < m_numComponents; ++i) {
            stddevs(i) = std::sqrt(m_variances(i));
        }
        return stddevs;
    }
    
    T getTotalVariance() const { return m_totalVariance; }
    size_t getNumComponents() const { return m_numComponents; }
    size_t getNumDimensions() const { return m_numDimensions; }
    size_t getNumSamples() const { return m_numSamples; }
    bool isFitted() const { return m_fitted; }
    
    /**
     * @brief Saves the PCA model to binary file
     * 
     * File format:
     * - 4 bytes: magic "PCA1"
     * - 8 bytes: numDimensions
     * - 8 bytes: numSamples  
     * - 8 bytes: numComponents
     * - 8 bytes: totalVariance
     * - D*8 bytes: mean
     * - K*8 bytes: variances
     * - D*K*8 bytes: components (column-major)
     * 
     * @param filename Output file path
     */
    void save(const std::string& filename) const {
        if (!m_fitted) {
            throw std::runtime_error("Cannot save unfitted model");
        }
        
        std::ofstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for writing: " + filename);
        }
        
        // Write magic number
        const char magic[4] = {'P', 'C', 'A', '1'};
        file.write(magic, 4);
        
        // Write dimensions
        file.write(reinterpret_cast<const char*>(&m_numDimensions), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&m_numSamples), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&m_numComponents), sizeof(size_t));
        file.write(reinterpret_cast<const char*>(&m_totalVariance), sizeof(T));
        
        // Write mean
        for (size_t i = 0; i < m_numDimensions; ++i) {
            T val = m_mean(i);
            file.write(reinterpret_cast<const char*>(&val), sizeof(T));
        }
        
        // Write variances
        for (size_t i = 0; i < m_numComponents; ++i) {
            T val = m_variances(i);
            file.write(reinterpret_cast<const char*>(&val), sizeof(T));
        }
        
        // Write components (column-major)
        for (size_t j = 0; j < m_numComponents; ++j) {
            for (size_t i = 0; i < m_numDimensions; ++i) {
                T val = m_components(i, j);
                file.write(reinterpret_cast<const char*>(&val), sizeof(T));
            }
        }
        
        file.close();
        std::cout << "PCA model saved to " << filename << std::endl;
    }
    
    /**
     * @brief Loads a PCA model from binary file
     * @param filename Input file path
     */
    void load(const std::string& filename) {
        std::ifstream file(filename, std::ios::binary);
        if (!file) {
            throw std::runtime_error("Cannot open file for reading: " + filename);
        }
        
        // Read and verify magic number
        char magic[4];
        file.read(magic, 4);
        if (magic[0] != 'P' || magic[1] != 'C' || magic[2] != 'A' || magic[3] != '1') {
            throw std::runtime_error("Invalid PCA file format");
        }
        
        // Read dimensions
        file.read(reinterpret_cast<char*>(&m_numDimensions), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&m_numSamples), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&m_numComponents), sizeof(size_t));
        file.read(reinterpret_cast<char*>(&m_totalVariance), sizeof(T));
        
        // Read mean
        m_mean = Vector<T>(m_numDimensions);
        for (size_t i = 0; i < m_numDimensions; ++i) {
            T val;
            file.read(reinterpret_cast<char*>(&val), sizeof(T));
            m_mean(i) = val;
        }
        
        // Read variances
        m_variances = Vector<T>(m_numComponents);
        for (size_t i = 0; i < m_numComponents; ++i) {
            T val;
            file.read(reinterpret_cast<char*>(&val), sizeof(T));
            m_variances(i) = val;
        }
        
        // Read components
        m_components = Matrix2D<T>(m_numDimensions, m_numComponents);
        for (size_t j = 0; j < m_numComponents; ++j) {
            for (size_t i = 0; i < m_numDimensions; ++i) {
                T val;
                file.read(reinterpret_cast<char*>(&val), sizeof(T));
                m_components(i, j) = val;
            }
        }
        
        file.close();
        m_fitted = true;
        
        std::cout << "PCA model loaded: D=" << m_numDimensions 
                  << ", N=" << m_numSamples 
                  << ", K=" << m_numComponents << std::endl;
    }
    
    /**
     * @brief Prints a summary of the PCA model
     */
    void printSummary() const {
        if (!m_fitted) {
            std::cout << "PCA model not fitted" << std::endl;
            return;
        }
        
        std::cout << "\n========== PCA Model Summary ==========" << std::endl;
        std::cout << "Dimensions (D):     " << m_numDimensions << std::endl;
        std::cout << "Training samples:   " << m_numSamples << std::endl;
        std::cout << "Components (K):     " << m_numComponents << std::endl;
        std::cout << "Total variance:     " << m_totalVariance << std::endl;
        
        auto ratios = explainedVarianceRatio();
        auto cumulative = cumulativeVarianceRatio();
        
        std::cout << "\nVariance explained by component:" << std::endl;
        std::cout << std::fixed << std::setprecision(4);
        
        size_t displayCount = std::min(m_numComponents, static_cast<size_t>(10));
        for (size_t i = 0; i < displayCount; ++i) {
            std::cout << "  PC" << std::setw(2) << (i+1) << ": " 
                      << std::setw(7) << (ratios(i) * 100) << "%  "
                      << "(cumulative: " << std::setw(7) << (cumulative(i) * 100) << "%)"
                      << std::endl;
        }
        
        if (m_numComponents > 10) {
            std::cout << "  ... (" << (m_numComponents - 10) << " more components)" << std::endl;
        }
        
        std::cout << "\nComponents needed for variance thresholds:" << std::endl;
        std::cout << "  90%: " << componentsForVariance(0.90) << " components" << std::endl;
        std::cout << "  95%: " << componentsForVariance(0.95) << " components" << std::endl;
        std::cout << "  99%: " << componentsForVariance(0.99) << " components" << std::endl;
        std::cout << "========================================\n" << std::endl;
    }
};

#endif // PCA_HPP
