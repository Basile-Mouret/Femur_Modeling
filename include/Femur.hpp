#ifndef FEMUR_HPP
#define FEMUR_HPP

#include <string>
#include "linalg.hpp"

class Femur{
    private:
        Matrix2D m_coords;
        Matrix2D m_normals;
        Matrix2D m_triangles;

    public:
        //Constructors
        /*
        * @brief Constructs a default Femur
        */
        Femur();

        /*
        * @brief Constructs a Femur from a obj file
        * @param filename : std::string
        */
        Femur(std::string filename);


        /*
        * @brief Constructs a Femur from data
        * @param coords : Matrix2D
        * @param normals : Matrix2D
        * @param triangles : Matrix2D
        */
        Femur(Matrix2D coords, Matrix2D normals, Matrix2D triangles);
        
        // Saving
        /*
        * @brief Saves a Femur in an obj file
        * @param coords : Matrix2D
        */
        void saveToFile(std::string filepath) const;

        // getters
        /*
        * @brief Gets the coordinate Matrix
        */
        Matrix2D getCoords() const;

        /*
        * @brief Gets the normals Matrix
        */
        Matrix2D getNormals() const;

        /*
        * @brief Gets the triangles Matrix
        */
        Matrix2D getTriangles() const;
};

#endif

