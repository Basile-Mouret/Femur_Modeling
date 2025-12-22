#include "linalg.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>


// Vecor class method implementations
std::ostream& operator<<(std::ostream& os, const Vector& vec) {
    os << "[";
    for(size_t i = 0; i < vec.m_size; ++i) {
        os << vec.m_data(i);
        if(i != vec.m_size - 1) {
            os << ", ";
        }
    }
    os << "]";
    return os;
}


//Constructors
Vector::Vector(size_t s) : m_size(s), m_data(Eigen::VectorXd::Zero(s)) {}

Vector::Vector(size_t s, double init_value) : m_size(s), m_data(Eigen::VectorXd::Constant(s, init_value)) {}

Vector::Vector(size_t s, const std::vector<double>& init_values) : m_size(s), m_data(Eigen::VectorXd::Zero(s)) {
    for(size_t i = 0; i < s && i < init_values.size(); ++i) {
        m_data(i) = init_values[i];
    }
}

Vector::Vector(const Vector& other) : m_size(other.m_size), m_data(other.m_data) {}


// Basic operations
size_t Vector::getSize() const {
    return m_size;
}

bool Vector::isZero() const {
    for(size_t i = 0; i < m_size; ++i) {
        if(m_data(i) != 0) {
            return false;
        }
    }
    return true;
}

bool Vector::operator==(const Vector& other) const {
    if(m_size != other.m_size) {
        return false;
    }
    for (size_t i = 0; i < m_size; ++i) {
        if(m_data(i) != other.m_data(i)) {
            return false;
        }
    }
    return true;
}

double Vector::operator()(size_t i_index) const {
    return m_data(i_index);
}

double &Vector::operator()(size_t i_index) {
    return m_data(i_index);
}

bool Vector::setCoeff(size_t i_index, double value) {
    if(i_index >= m_size) {
        return false;
    }
    m_data(i_index) = value;
    return true;
}


// Linear algebra operations

Vector Vector::operator*(const double scalar){
    Vector result(m_size);
    for (size_t i = 0; i < m_size; ++i) {
        result.setCoeff(i, m_data(i) * scalar);
    }
    return result;
}

Vector Vector::operator+(const Vector &other){
    if (m_size != other.m_size) {
        std::cout << "ERROR: Vectors must be of the same size for addition." << std::endl;
        return *this;
    }
    Vector result(m_size);
    for (size_t i = 0; i < m_size; ++i) {
        result.setCoeff(i, m_data(i) + other.m_data(i));
    }
    return result;
}

Vector Vector::operator-(const Vector &other){
    if (m_size != other.m_size) {
        std::cout << "ERROR: Vectors must be of the same size for subtraction." << std::endl;
        return *this;
    }
    Vector result(m_size);
    for (size_t i = 0; i < m_size; ++i) {
        result.setCoeff(i, m_data(i) - other.m_data(i));
    }
    return result;
}

double Vector::dot(const Vector& other){
    if(m_size != other.m_size) {
        std::cout << "ERROR: Vectors must be of the same size for dot product." << std::endl;
        return 0.0f;
    }
    double result = 0;
    for(size_t i = 0; i < m_size; ++i) {
        result += m_data(i) * other.m_data(i);
    }
    return result;    
}




// Matrix2D class method implementations
std::ostream& operator<<(std::ostream& os, const Matrix2D& mat) {
    os << "[";
    for(size_t i = 0; i < mat.m_rows; ++i) {
        os << "[";
        for(size_t j = 0; j < mat.m_cols; ++j) {
            os << mat.m_data(i, j);
            if(j != mat.m_cols - 1) {
                os << ", ";
            }
        }
        os << "]";
        if(i != mat.m_rows - 1) {
            os << ",\n ";
        }
    }
    os << "]";
    return os;
}

//Constructors
Matrix2D::Matrix2D(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_data(Eigen::MatrixXd::Zero(rows, cols)) {}

Matrix2D::Matrix2D(size_t rows, size_t cols, const double init_value) : m_rows(rows), m_cols(cols), m_data(Eigen::MatrixXd::Constant(rows, cols, init_value)) {}

Matrix2D::Matrix2D(const Matrix2D& other) : m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data) {}


// Basic operations
size_t Matrix2D::getSizeRows() const {
    return m_rows;
}

size_t Matrix2D::getSizeCols() const {
    return m_cols;
}

Vector Matrix2D::getRow(size_t i_row) const {
    if(i_row >= m_rows) {
        std::cout << "ERROR: Row index out of bounds." << std::endl;
        return Vector(m_cols);
    }
    Vector rowVec(m_cols);
    for(size_t j = 0; j < m_cols; ++j) {
        rowVec.setCoeff(j, m_data(i_row, j));
    }
    return rowVec;
}

Vector Matrix2D::getCol(size_t i_col) const {
    if(i_col >= m_cols) {
        std::cout << "ERROR: Column index out of bounds." << std::endl;
        return Vector(m_rows);
    }
    Vector colVec(m_rows);
    for(size_t i = 0; i < m_rows; ++i) {
        colVec.setCoeff(i, m_data(i, i_col));
    }
    return colVec;
}

bool Matrix2D::setCoeff(size_t i_row, size_t i_col, double value) {
    if(i_row >= m_rows || i_col >= m_cols) {
        std::cout << "ERROR: Index out of bounds." << std::endl;
        return false;
    }
    m_data(i_row, i_col) = value;
    return true;
}

bool Matrix2D::setRow(size_t i_row, const Vector& row) {
    if(i_row >= m_rows || row.getSize() != m_cols) {
        std::cout << "ERROR: Row index out of bounds or size mismatch." << std::endl;
        return false;
    }
    for(size_t j = 0; j < m_cols; ++j) {
        m_data(i_row, j) = row(j);
    }
    return true;
}

bool Matrix2D::setCol(size_t i_col, const Vector& col) {
    if(i_col >= m_cols || col.getSize() != m_rows) {
        std::cout << "ERROR: Column index out of bounds or size mismatch." << std::endl;
        return false;
    }
    for(size_t i = 0; i < m_rows; ++i) {
        m_data(i, i_col) = col(i);
    }
    return true;
}

bool Matrix2D::isZero() const {
    for(size_t i = 0; i < m_rows; ++i) {
        for(size_t j = 0; j < m_cols; ++j) {
            if(m_data(i, j) != 0) {
                return false;
            }
        }
    }
    return true;
}

bool Matrix2D::operator==(const Matrix2D& other) const {
    if(m_rows != other.m_rows || m_cols != other.m_cols) {
        return false;
    }
    for(size_t i = 0; i < m_rows; ++i) {
        for(size_t j = 0; j < m_cols; ++j) {
            if(m_data(i, j) != other.m_data(i, j)) {
                return false;
            }
        }
    }
    return true;
}

double& Matrix2D::operator()(size_t i_row, size_t i_col) {
    return m_data(i_row, i_col);
}

double Matrix2D::operator()(size_t i_row, size_t i_col) const {
    return m_data(i_row, i_col);
}

// Linear algebra operations
Matrix2D Matrix2D::operator*(const double scalar){
    Matrix2D result(m_rows, m_cols);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < m_cols; ++j) {
            result.setCoeff(i, j, m_data(i, j) * scalar);
        }
    }
    return result;
}

Matrix2D Matrix2D::operator+(const  Matrix2D &other){
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        std::cout << "ERROR: Matrices must be of the same size for addition." << std::endl;
        return *this;
    }
    Matrix2D result(m_rows, m_cols);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < m_cols; ++j) {
            result.setCoeff(i, j, m_data(i, j) + other.m_data(i, j));
        }
    }
    return result;
}

Matrix2D Matrix2D::operator-(const  Matrix2D &other){
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        std::cout << "ERROR: Matrices must be of the same size for subtraction." << std::endl;
        return *this;
    }
    Matrix2D result(m_rows, m_cols);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < m_cols; ++j) {
            result.setCoeff(i, j, m_data(i, j) - other.m_data(i, j));
        }
    }
    return result;
}

Matrix2D Matrix2D::operator*(const Matrix2D &other){
    if (m_cols != other.m_rows){
        std::cout << "ERROR: Matrix A columns must match Matrix B rows for multiplication." << std::endl;
        return Matrix2D(0, 0);
    }
    Matrix2D result(m_rows, other.m_cols);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < other.m_cols; ++j) {
            double sum = 0;
            for (size_t k = 0; k < m_cols; ++k) {
                sum += m_data(i, k) * other.m_data(k, j);
            }
            result.setCoeff(i, j, sum);
        }
    }
    return result;
}

Vector Matrix2D::operator*(const Vector &vec){
    if (m_cols != vec.getSize()){
        std::cout << "ERROR: Matrix columns must match vector size for multiplication." << std::endl;
        return Vector(0);
    }
    Vector result(m_rows);
    for (size_t j = 0; j < m_rows; ++j) {
        double sum = 0;
        for (size_t i = 0; i < m_cols; ++i) {
            sum += m_data(j, i) * vec(i);
        }
        result.setCoeff(j, sum);
    }
    return result;
}


// Matrix2dSquare class method implementations

//Constructors
Matrix2DSquare::Matrix2DSquare(size_t size) : Matrix2D(size, size) {}

Matrix2DSquare::Matrix2DSquare(size_t size, const double init_value) : Matrix2D(size, size, init_value) {}

Matrix2DSquare::Matrix2DSquare(const Matrix2DSquare& other) : Matrix2D(other) {}
