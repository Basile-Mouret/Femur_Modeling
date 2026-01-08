/**
 * @file Femur.hpp
 * @brief Declaration of the Femur class for 3D femur representation
 * @details This file contains the Femur class which represents a 3D femur model
 *          using coordinates, normals, and triangle faces. Supports loading from
 *          and saving to OBJ file format.
 */

#ifndef FEMUR_HPP
#define FEMUR_HPP

#include <string>
#include "linalg.hpp"

/**
 * @class Femur
 * @brief Represents a 3D femur model with geometric data
 * 
 * The Femur class stores and manages 3D geometric data for a femur bone,
 * including vertex coordinates, surface normals, and triangular faces.
 * It provides functionality to load from and save to OBJ file format.
 */
class Femur{
    private:
        Matrix2D<double> m_coords;      ///< Vertex coordinates matrix (N x 3)
        Matrix2D<double> m_normals;     ///< Normal vectors matrix (N x 3)
        Matrix2D<double> m_triangles;   ///< Triangle indices matrix (M x 3)

    public:
        // Constructors
        
        /**
         * @brief Constructs a default empty Femur object
         * 
         * Creates a Femur instance with empty coordinate, normal, and triangle data.
         */
        Femur();

        /**
         * @brief Constructs a Femur object from an OBJ file
         * 
         * Loads femur geometry data from a Wavefront OBJ file including
         * vertex coordinates, normals, and face definitions.
         * 
         * @param filename Path to the OBJ file to load
         */
        Femur(std::string filename);


        /**
         * @brief Constructs a Femur object from explicit geometric data
         * 
         * Creates a Femur instance with provided coordinate, normal, and triangle data.
         * 
         * @param coords Vertex coordinates matrix (N x 3)
         * @param normals Normal vectors matrix (N x 3)
         * @param triangles Triangle indices matrix (M x 3)
         */
        Femur(Matrix2D<double> coords, Matrix2D<double> normals, Matrix2D<double> triangles);
        
        // Saving
        
        /**
         * @brief Saves the Femur geometry to an OBJ file
         * 
         * Exports the femur's coordinates, normals, and triangular faces
         * to a Wavefront OBJ file format.
         * 
         * @param filepath Path where the OBJ file will be saved
         */
        void saveToFile(std::string filepath) const;

        // Getters
        
        /**
         * @brief Gets the vertex coordinates matrix
         * 
         * @return Matrix2D<double> containing vertex coordinates (N x 3)
         */
        Matrix2D<double> getCoords() const;

        /**
         * @brief Gets the normal vectors matrix
         * 
         * @return Matrix2D<double> containing normal vectors (N x 3)
         */
        Matrix2D<double> getNormals() const;

        /**
         * @brief Gets the triangle indices matrix
         * 
         * @return Matrix2D<double> containing triangle face indices (M x 3)
         */
        Matrix2D<double> getTriangles() const;

        /**
         * @brief Gets the vertex coordinates as a vector 
         * 
         * @return Vector<double> containing triangle face indices (3M) (stacked columns)
         */
        template<typename T>
        Vector<T> getCoordsVect() const;

};

#endif

