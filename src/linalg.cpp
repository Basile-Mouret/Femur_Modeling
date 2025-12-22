#include "linalg.hpp"
#include <Eigen/Dense>
#include <iostream>
#include <vector>


// Vector class method implementations
template<typename T>
std::ostream& operator<<(std::ostream& os, const Vector<T>& vec) {
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
template<typename T>
Vector<T>::Vector(size_t s) : m_size(s), m_data(Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(s)) {}

template<typename T>
Vector<T>::Vector(size_t s, T init_value) : m_size(s), m_data(Eigen::Matrix<T, Eigen::Dynamic, 1>::Constant(s, init_value)) {}

template<typename T>
Vector<T>::Vector(size_t s, const std::vector<T>& init_values) : m_size(s), m_data(Eigen::Matrix<T, Eigen::Dynamic, 1>::Zero(s)) {
    for(size_t i = 0; i < s && i < init_values.size(); ++i) {
        m_data(i) = init_values[i];
    }
}

template<typename T>
Vector<T>::Vector(const Vector<T>& other) : m_size(other.m_size), m_data(other.m_data) {}


// Basic operations
template<typename T>
size_t Vector<T>::getSize() const {
    return m_size;
}

template<typename T>
bool Vector<T>::isZero() const {
    for(size_t i = 0; i < m_size; ++i) {
        if(m_data(i) != 0) {
            return false;
        }
    }
    return true;
}

template<typename T>
bool Vector<T>::operator==(const Vector<T>& other) const {
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

template<typename T>
T Vector<T>::operator()(size_t i_index) const {
    return m_data(i_index);
}

template<typename T>
bool Vector<T>::setCoeff(size_t i_index, T value) {
    if(i_index >= m_size) {
        return false;
    }
    m_data(i_index) = value;
    return true;
}


// Linear algebra operations

template<typename T>
Vector<T> Vector<T>::operator*(const T scalar){
    Vector<T> result(m_size);
    for (size_t i = 0; i < m_size; ++i) {
        result.setCoeff(i, m_data(i) * scalar);
    }
    return result;
}

template<typename T>
Vector<T> Vector<T>::operator+(const Vector<T> &other){
    if (m_size != other.m_size) {
        std::cout << "ERROR: Vectors must be of the same size for addition." << std::endl;
        return *this;
    }
    Vector<T> result(m_size);
    for (size_t i = 0; i < m_size; ++i) {
        result.setCoeff(i, m_data(i) + other.m_data(i));
    }
    return result;
}

template<typename T>
Vector<T> Vector<T>::operator-(const Vector<T> &other){
    if (m_size != other.m_size) {
        std::cout << "ERROR: Vectors must be of the same size for subtraction." << std::endl;
        return *this;
    }
    Vector<T> result(m_size);
    for (size_t i = 0; i < m_size; ++i) {
        result.setCoeff(i, m_data(i) - other.m_data(i));
    }
    return result;
}

template<typename T>
T Vector<T>::dot(const Vector<T>& other){
    if(m_size != other.m_size) {
        std::cout << "ERROR: Vectors must be of the same size for dot product." << std::endl;
        return T(0);
    }
    T result = 0;
    for(size_t i = 0; i < m_size; ++i) {
        result += m_data(i) * other.m_data(i);
    }
    return result;    
}




// Matrix2D class method implementations
template<typename T>
std::ostream& operator<<(std::ostream& os, const Matrix2D<T>& mat) {
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
template<typename T>
Matrix2D<T>::Matrix2D(size_t rows, size_t cols) : m_rows(rows), m_cols(cols), m_data(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Zero(rows, cols)) {}

template<typename T>
Matrix2D<T>::Matrix2D(size_t rows, size_t cols, const T init_value) : m_rows(rows), m_cols(cols), m_data(Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>::Constant(rows, cols, init_value)) {}

template<typename T>
Matrix2D<T>::Matrix2D(const Matrix2D<T>& other) : m_rows(other.m_rows), m_cols(other.m_cols), m_data(other.m_data) {}


// Basic operations
template<typename T>
size_t Matrix2D<T>::getSizeRows() const {
    return m_rows;
}

template<typename T>
size_t Matrix2D<T>::getSizeCols() const {
    return m_cols;
}

template<typename T>
Vector<T> Matrix2D<T>::getRow(size_t i_row) const {
    if(i_row >= m_rows) {
        std::cout << "ERROR: Row index out of bounds." << std::endl;
        return Vector<T>(m_cols);
    }
    Vector<T> rowVec(m_cols);
    for(size_t j = 0; j < m_cols; ++j) {
        rowVec.setCoeff(j, m_data(i_row, j));
    }
    return rowVec;
}

template<typename T>
Vector<T> Matrix2D<T>::getCol(size_t i_col) const {
    if(i_col >= m_cols) {
        std::cout << "ERROR: Column index out of bounds." << std::endl;
        return Vector<T>(m_rows);
    }
    Vector<T> colVec(m_rows);
    for(size_t i = 0; i < m_rows; ++i) {
        colVec.setCoeff(i, m_data(i, i_col));
    }
    return colVec;
}

template<typename T>
bool Matrix2D<T>::setCoeff(size_t i_row, size_t i_col, T value) {
    if(i_row >= m_rows || i_col >= m_cols) {
        std::cout << "ERROR: Index out of bounds." << std::endl;
        return false;
    }
    m_data(i_row, i_col) = value;
    return true;
}

template<typename T>
bool Matrix2D<T>::setRow(size_t i_row, const Vector<T>& row) {
    if(i_row >= m_rows || row.getSize() != m_cols) {
        std::cout << "ERROR: Row index out of bounds or size mismatch." << std::endl;
        return false;
    }
    for(size_t j = 0; j < m_cols; ++j) {
        m_data(i_row, j) = row(j);
    }
    return true;
}

template<typename T>
bool Matrix2D<T>::setCol(size_t i_col, const Vector<T>& col) {
    if(i_col >= m_cols || col.getSize() != m_rows) {
        std::cout << "ERROR: Column index out of bounds or size mismatch." << std::endl;
        return false;
    }
    for(size_t i = 0; i < m_rows; ++i) {
        m_data(i, i_col) = col(i);
    }
    return true;
}

template<typename T>
bool Matrix2D<T>::isZero() const {
    for(size_t i = 0; i < m_rows; ++i) {
        for(size_t j = 0; j < m_cols; ++j) {
            if(m_data(i, j) != 0) {
                return false;
            }
        }
    }
    return true;
}

template<typename T>
bool Matrix2D<T>::operator==(const Matrix2D<T>& other) const {
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

template<typename T>
T Matrix2D<T>::operator()(size_t i_row, size_t i_col) const {
    return m_data(i_row, i_col);
}

// Linear algebra operations
template<typename T>
Matrix2D<T> Matrix2D<T>::operator*(const float scalar){
    Matrix2D<T> result(m_rows, m_cols);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < m_cols; ++j) {
            result.setCoeff(i, j, m_data(i, j) * scalar);
        }
    }
    return result;
}

template<typename T>
Matrix2D<T> Matrix2D<T>::operator+(const  Matrix2D<T> &other){
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        std::cout << "ERROR: Matrices must be of the same size for addition." << std::endl;
        return *this;
    }
    Matrix2D<T> result(m_rows, m_cols);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < m_cols; ++j) {
            result.setCoeff(i, j, m_data(i, j) + other.m_data(i, j));
        }
    }
    return result;
}

template<typename T>
Matrix2D<T> Matrix2D<T>::operator-(const  Matrix2D<T> &other){
    if (m_rows != other.m_rows || m_cols != other.m_cols) {
        std::cout << "ERROR: Matrices must be of the same size for subtraction." << std::endl;
        return *this;
    }
    Matrix2D<T> result(m_rows, m_cols);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < m_cols; ++j) {
            result.setCoeff(i, j, m_data(i, j) - other.m_data(i, j));
        }
    }
    return result;
}

template<typename T>
Matrix2D<T> Matrix2D<T>::operator*(const Matrix2D<T> &other){
    if (m_cols != other.m_rows){
        std::cout << "ERROR: Matrix A columns must match Matrix B rows for multiplication." << std::endl;
        return Matrix2D<T>(0, 0);
    }
    Matrix2D<T> result(m_rows, other.m_cols);
    for (size_t i = 0; i < m_rows; ++i) {
        for (size_t j = 0; j < other.m_cols; ++j) {
            T sum = 0;
            for (size_t k = 0; k < m_cols; ++k) {
                sum += m_data(i, k) * other.m_data(k, j);
            }
            result.setCoeff(i, j, sum);
        }
    }
    return result;
}

template<typename T>
Vector<T> Matrix2D<T>::operator*(const Vector<T> &vec){
    if (m_cols != vec.getSize()){
        std::cout << "ERROR: Matrix columns must match vector size for multiplication." << std::endl;
        return Vector<T>(0);
    }
    Vector<T> result(m_rows);
    for (size_t j = 0; j < m_rows; ++j) {
        T sum = 0;
        for (size_t i = 0; i < m_cols; ++i) {
            sum += m_data(j, i) * vec(i);
        }
        result.setCoeff(j, sum);
    }
    return result;
}


// Matrix2DSquare class method implementations

//Constructors
template<typename T>
Matrix2DSquare<T>::Matrix2DSquare(size_t size) : Matrix2D<T>(size, size) {}

template<typename T>
Matrix2DSquare<T>::Matrix2DSquare(size_t size, const T init_value) : Matrix2D<T>(size, size, init_value) {}

template<typename T>
Matrix2DSquare<T>::Matrix2DSquare(const Matrix2DSquare<T>& other) : Matrix2D<T>(other) {}


// ============================================================================
// Explicit template instantiations
// ============================================================================

template class Vector<float>;
template class Vector<double>;
template class Vector<int>;
template class Vector<long>;
template class Vector<short>;
template class Vector<unsigned int>;
template class Vector<unsigned long>;

template class Matrix2D<float>;
template class Matrix2D<double>;
template class Matrix2D<int>;
template class Matrix2D<long>;
template class Matrix2D<short>;
template class Matrix2D<unsigned int>;
template class Matrix2D<unsigned long>;

template class Matrix2DSquare<float>;
template class Matrix2DSquare<double>;
template class Matrix2DSquare<int>;
template class Matrix2DSquare<long>;
template class Matrix2DSquare<short>;
template class Matrix2DSquare<unsigned int>;
template class Matrix2DSquare<unsigned long>;

template std::ostream& operator<<(std::ostream& os, const Vector<float>& vec);
template std::ostream& operator<<(std::ostream& os, const Vector<double>& vec);
template std::ostream& operator<<(std::ostream& os, const Vector<int>& vec);
template std::ostream& operator<<(std::ostream& os, const Vector<long>& vec);
template std::ostream& operator<<(std::ostream& os, const Vector<short>& vec);
template std::ostream& operator<<(std::ostream& os, const Vector<unsigned int>& vec);
template std::ostream& operator<<(std::ostream& os, const Vector<unsigned long>& vec);

template std::ostream& operator<<(std::ostream& os, const Matrix2D<float>& mat);
template std::ostream& operator<<(std::ostream& os, const Matrix2D<double>& mat);
template std::ostream& operator<<(std::ostream& os, const Matrix2D<int>& mat);
template std::ostream& operator<<(std::ostream& os, const Matrix2D<long>& mat);
template std::ostream& operator<<(std::ostream& os, const Matrix2D<short>& mat);
template std::ostream& operator<<(std::ostream& os, const Matrix2D<unsigned int>& mat);
template std::ostream& operator<<(std::ostream& os, const Matrix2D<unsigned long>& mat);
