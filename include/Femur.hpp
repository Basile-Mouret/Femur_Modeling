#ifndef FEMUR_HPP
#define FEMUR_HPP

#include <string>
#include "linalg.hpp"

class Femur{
    private:
        Matrix2D<double> m_coords;
        Matrix2D<double> m_normals;
        Matrix2D<double> m_triangles;

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
        * @param coords : Matrix2D<double>
        * @param normals : Matrix2D<double>
        * @param triangles : Matrix2D<double>
        */
        Femur(Matrix2D<double> coords, Matrix2D<double> normals, Matrix2D<double> triangles);
        
        // Saving
        /*
        * @brief Saves a Femur in an obj file
        * @param coords : Matrix2D<double>
        */
        void saveToFile(std::string filepath) const;

        // getters
        /*
        * @brief Gets the coordinate Matrix
        */
        Matrix2D<double> getCoords() const;

        /*
        * @brief Gets the normals Matrix
        */
        Matrix2D<double> getNormals() const;

        /*
        * @brief Gets the triangles Matrix
        */
        Matrix2D<double> getTriangles() const;
};

#endif

