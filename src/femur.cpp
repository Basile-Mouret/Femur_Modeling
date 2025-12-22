#include <fstream>
#include <string>
#include <iomanip>
#include <limits>

#include "Femur.hpp"
#include "linalg.hpp"

Femur::Femur() : m_coords(18291,3), m_normals(18291,3), m_triangles(36578,3){}

Femur::Femur(Matrix2D<double> coords, Matrix2D<double> normals, Matrix2D<double> triangles) : m_coords(coords), m_normals(normals), m_triangles(triangles){}

Femur::Femur(std::string filename) : m_coords(18291,3), m_normals(18291,3), m_triangles(36578,3){

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "CRITICAL ERROR: Could not open file: " << filename << std::endl;
        std::cerr << "Check your path! Current working dir is likely 'build/'" << std::endl;
        return;
    }

    std::string line;

    std::getline(file, line); // # 18291 vertice(s)
    for (size_t i=0; i<18291; i++){
        std::getline(file, line);
        sscanf(line.c_str(), "v %lf %lf %lf", &m_coords(i,0), &m_coords(i,1), &m_coords(i,2));
    }
    std::getline(file, line); // empty line 
    std::getline(file, line); // # 18291 normal(s)

    for (size_t i=0; i<18291; i++){
        std::getline(file, line);
        sscanf(line.c_str(), "vn %lf %lf %lf", &m_normals(i,0), &m_normals(i,1), &m_normals(i,2));
    }

    std::getline(file, line); // empty line 
    std::getline(file, line); // # 36578 triangle(s)

    for (size_t i=0; i<36578; i++){
        std::getline(file, line);
        int v1, v2, v3, n1, n2, n3;
        sscanf(line.c_str(), "f %d//%d %d//%d %d//%d", &v1, &n1, &v2, &n2, &v3, &n3);
        m_triangles(i,0) = v1-1;
        m_triangles(i,1) = v2-1;
        m_triangles(i,2) = v3-1;
    }
        
    file.close();
}

void Femur::saveToFile(std::string filepath) const {
    std::ofstream file(filepath);
    
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file for writing: " << filepath << std::endl;
        return;
    }

    // sets the number of digits needed for maximum precision
    file << std::setprecision(15);
    
    //drop trailing zeros.
    file << std::defaultfloat;

    // upper case e's
    file << std::uppercase;

    file << "# " << m_coords.getSizeRows() << " vertice(s)\r\n";
    // Write Vertices (v x y z)
    for (int i = 0; i < m_coords.getSizeRows(); ++i) {
        file << "v " << m_coords(i, 0) << " " 
                     << m_coords(i, 1) << " " 
                     << m_coords(i, 2) << "\r\n";
    }
    file << "\r\n";
    file << "# " << m_normals.getSizeRows() << " normal(s)\r\n";

    // Write Normals (vn x y z)
    for (int i = 0; i < m_normals.getSizeRows(); ++i) {
        file << "vn " << m_normals(i, 0) << " " 
                      << m_normals(i, 1) << " " 
                      << m_normals(i, 2) << "\r\n";
    }
    file << "\r\n";
    file << "# " << m_triangles.getSizeRows() << " triangle(s)\r\n";
    // Write Faces (f v1//n1 v2//n2 v3//n3)
    for (int i = 0; i < m_triangles.getSizeRows(); ++i) {
        int v1 = static_cast<int>(m_triangles(i, 0)) + 1;
        int v2 = static_cast<int>(m_triangles(i, 1)) + 1;
        int v3 = static_cast<int>(m_triangles(i, 2)) + 1;

        file << "f " << v1 << "//" << v1 << " " 
                     << v2 << "//" << v2 << " " 
                     << v3 << "//" << v3 << "\r\n";
    }

    file.close();
    std::cout << "Successfully saved to " << filepath << std::endl;
}

Matrix2D<double> Femur::getCoords() const{
   return m_coords;
}
Matrix2D<double> Femur::getNormals() const{
   return m_normals;
}

Matrix2D<double> Femur::getTriangles() const{
   return m_triangles;
}

