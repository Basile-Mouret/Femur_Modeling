/**
 * @file dataset.hpp
 * @brief Dataset utilities for loading and managing femur shape data
 * @author Femur Modeling Project
 * @date 2024
 * 
 * This file provides utilities for loading multiple femur shapes from
 * OBJ files and converting them into matrix format suitable for PCA.
 */

#ifndef DATASET_HPP
#define DATASET_HPP

#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <sstream>
#include <stdexcept>

#include "femur.hpp"
#include "linalg.hpp"

/**
 * @class FemurDataset
 * @brief Manages a collection of femur shapes for statistical analysis
 * 
 * This class handles loading multiple femur OBJ files from a directory
 * and converting them to matrix format for PCA analysis.
 * 
 * @tparam T Numeric type (float or double)
 * 
 * Data Format:
 * Each femur is converted to a vector of length D = 3 * num_vertices.
 * The standardized coordinates are stored in "stacked" format:
 *   [x0, x1, ..., xN, y0, y1, ..., yN, z0, z1, ..., zN]
 * 
 * This matches the standardization in femur.cpp where coordinates are
 * divided by [152.2, 20.4, 16.2] respectively.
 */
template <typename T>
class FemurDataset {
private:
    std::vector<Femur> m_femurs;           ///< Collection of loaded femurs
    std::vector<std::string> m_filenames;   ///< Filenames for each femur
    size_t m_numVertices;                   ///< Number of vertices per femur
    bool m_standardized;                    ///< Whether data is standardized
    unsigned int m_sampleRate;              ///< Vertex sampling rate
    bool m_loaded;                          ///< Whether data has been loaded

public:
    /**
     * @brief Default constructor
     */
    FemurDataset() : m_numVertices(0), m_standardized(true), m_sampleRate(1), m_loaded(false) {}
    
    /**
     * @brief Constructor that loads from directory
     * @param directory Path to directory containing OBJ files
     * @param standardized Whether to use standardized coordinates
     * @param sampleRate Vertex sampling rate (1 = all vertices)
     */
    explicit FemurDataset(const std::string& directory, 
                          bool standardized = true,
                          unsigned int sampleRate = 1) 
        : m_numVertices(0), m_standardized(standardized), 
          m_sampleRate(sampleRate), m_loaded(false) {
        loadFromDirectory(directory, standardized, sampleRate);
    }
    
    /**
     * @brief Loads all OBJ files from a directory
     * 
     * @param directory Path to directory
     * @param standardized Whether to use standardized coordinates
     * @param sampleRate Vertex sampling rate
     * @return Number of files loaded
     */
    size_t loadFromDirectory(const std::string& directory, 
                             bool standardized = true,
                             unsigned int sampleRate = 1) {
        m_femurs.clear();
        m_filenames.clear();
        m_standardized = standardized;
        m_sampleRate = sampleRate;
        
        if (!std::filesystem::exists(directory)) {
            throw std::runtime_error("Directory does not exist: " + directory);
        }
        
        // Collect OBJ files
        std::vector<std::filesystem::path> files;
        for (const auto& entry : std::filesystem::directory_iterator(directory)) {
            if (entry.is_regular_file()) {
                std::string filename = entry.path().filename().string();
                if (filename.find(".obj") != std::string::npos) {
                    files.push_back(entry.path());
                }
            }
        }
        
        // Sort for consistent ordering
        std::sort(files.begin(), files.end());
        
        if (files.empty()) {
            throw std::runtime_error("No OBJ files found in: " + directory);
        }
        
        std::cout << "Loading " << files.size() << " femur files..." << std::endl;
        
        // Load each file
        for (const auto& filepath : files) {
            try {
                Femur femur(filepath.string());
                
                // Verify vertex count consistency
                size_t numVerts = femur.getCoords().getSizeRows();
                if (m_femurs.empty()) {
                    m_numVertices = numVerts;
                } else if (numVerts != m_numVertices) {
                    std::cerr << "Warning: Skipping " << filepath.filename() 
                              << " - vertex count mismatch ("
                              << numVerts << " vs " 
                              << m_numVertices << ")" << std::endl;
                    continue;
                }
                
                m_femurs.push_back(std::move(femur));
                m_filenames.push_back(filepath.filename().string());
                
            } catch (const std::exception& e) {
                std::cerr << "Warning: Failed to load " << filepath.filename() 
                          << ": " << e.what() << std::endl;
            }
        }
        
        if (m_femurs.empty()) {
            throw std::runtime_error("No valid femur files loaded");
        }
        
        m_loaded = true;
        std::cout << "Loaded " << m_femurs.size() << " femurs with " 
                  << m_numVertices << " vertices each" << std::endl;
        
        return m_femurs.size();
    }
    
    /**
     * @brief Adds a single femur to the dataset
     * @param femur Femur to add
     * @param filename Optional filename for reference
     */
    void addFemur(const Femur& femur, const std::string& filename = "") {
        size_t numVerts = femur.getCoords().getSizeRows();
        if (m_femurs.empty()) {
            m_numVertices = numVerts;
        } else if (numVerts != m_numVertices) {
            throw std::invalid_argument("Vertex count mismatch");
        }
        
        m_femurs.push_back(femur);
        m_filenames.push_back(filename);
        m_loaded = true;
    }
    
    /**
     * @brief Converts the dataset to a matrix for PCA
     * 
     * Creates a matrix where each column is a flattened femur shape.
     * Format: D x N where D = 3*num_vertices, N = num_femurs
     * 
     * The coordinates are in standardized form (from Femur::getCoordsVect())
     * and stored in stacked format: [all_X, all_Y, all_Z]
     * 
     * @return Data matrix (D x N)
     */
    Matrix2D<T> toMatrix() const {
        if (!m_loaded || m_femurs.empty()) {
            throw std::runtime_error("No data loaded");
        }
        
        // Get dimension from first shape
        Vector<T> firstVec = m_femurs[0].getCoordsVect<T>(m_sampleRate, m_standardized);
        size_t D = firstVec.getSize();
        size_t N = m_femurs.size();
        
        Matrix2D<T> data(D, N);
        
        for (size_t j = 0; j < N; ++j) {
            Vector<T> vec = m_femurs[j].getCoordsVect<T>(m_sampleRate, m_standardized);
            for (size_t i = 0; i < D; ++i) {
                data(i, j) = vec(i);
            }
        }
        
        return data;
    }
    
    /**
     * @brief Gets the shape vector for a specific femur
     * 
     * @param index Index of the femur
     * @return Shape vector in stacked format
     */
    Vector<T> getShapeVector(size_t index) const {
        if (index >= m_femurs.size()) {
            throw std::out_of_range("Femur index out of range");
        }
        return m_femurs[index].getCoordsVect<T>(m_sampleRate, m_standardized);
    }
    
    /**
     * @brief Converts a single femur to a vector
     * 
     * Uses standardized coordinates in stacked format:
     * [x0, x1, ..., xN, y0, y1, ..., yN, z0, z1, ..., zN]
     * 
     * @param femur Femur to convert
     * @return Flattened vector (D x 1)
     */
    Vector<T> femurToVector(const Femur& femur) const {
        return femur.getCoordsVect<T>(m_sampleRate, m_standardized);
    }
    
    /**
     * @brief Converts a vector back to a Femur object
     * 
     * @param vec Shape vector in stacked format (standardized)
     * @param templateFemur Femur to use as template for faces/normals
     * @return Reconstructed Femur object
     */
    Femur vectorToFemur(const Vector<T>& vec, const Femur& templateFemur) const {
        size_t n = vec.getSize() / 3;
        
        size_t templateVerts = templateFemur.getCoords().getSizeRows();
        if (n != templateVerts) {
            throw std::invalid_argument("Vector size doesn't match template");
        }
        
        // Create new femur from template and set coordinates
        Femur result(templateFemur.getCoords(), templateFemur.getNormals(), 
                     templateFemur.getTriangles());
        result.setCoordsVect<T>(vec, m_standardized);
        
        return result;
    }
    
    /**
     * @brief Loads a single shape from an OBJ file as a vector
     * 
     * Useful for loading test/validation shapes without adding
     * them to the dataset.
     * 
     * @param filename Path to OBJ file
     * @param standardized Whether to use standardized coordinates
     * @param sampleRate Vertex sampling rate
     * @return Shape vector in stacked format
     */
    static Vector<T> loadShapeFromFile(const std::string& filename,
                                       bool standardized = true,
                                       unsigned int sampleRate = 1) {
        Femur femur(filename);
        return femur.getCoordsVect<T>(sampleRate, standardized);
    }
    
    // Getters
    size_t size() const { return m_femurs.size(); }
    size_t getNumVertices() const { return m_numVertices; }
    size_t getDimension() const { return 3 * m_numVertices; }
    bool isLoaded() const { return m_loaded; }
    bool isStandardized() const { return m_standardized; }
    unsigned int getSampleRate() const { return m_sampleRate; }
    
    const Femur& getFemur(size_t index) const {
        if (index >= m_femurs.size()) {
            throw std::out_of_range("Femur index out of range");
        }
        return m_femurs[index];
    }
    
    const std::string& getFilename(size_t index) const {
        if (index >= m_filenames.size()) {
            throw std::out_of_range("Filename index out of range");
        }
        return m_filenames[index];
    }
    
    const std::vector<Femur>& getFemurs() const { return m_femurs; }
    const std::vector<std::string>& getFilenames() const { return m_filenames; }
    
    /**
     * @brief Prints dataset information
     */
    void printInfo() const {
        std::cout << "\n========== Dataset Info ==========" << std::endl;
        std::cout << "Number of shapes: " << m_femurs.size() << std::endl;
        std::cout << "Vertices per shape: " << m_numVertices << std::endl;
        std::cout << "Sample rate: " << m_sampleRate << std::endl;
        std::cout << "Standardized: " << (m_standardized ? "yes" : "no") << std::endl;
        
        // Calculate actual dimension with sample rate
        if (m_loaded && !m_femurs.empty()) {
            Vector<T> sample = m_femurs[0].getCoordsVect<T>(m_sampleRate, m_standardized);
            std::cout << "Dimension (D): " << sample.getSize() << std::endl;
        }
        
        if (!m_filenames.empty()) {
            std::cout << "\nLoaded files:" << std::endl;
            for (size_t i = 0; i < m_filenames.size(); ++i) {
                std::cout << "  " << (i+1) << ". " << m_filenames[i] << std::endl;
            }
        }
        std::cout << "=================================\n" << std::endl;
    }
};

#endif // DATASET_HPP
