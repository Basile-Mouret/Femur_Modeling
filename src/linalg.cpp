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

Vector::Vector(size_t s, float init_value) : m_size(s), m_data(Eigen::VectorXd::Constant(s, init_value)) {}

Vector::Vector(size_t s, const std::vector<float>& init_values) : m_size(s), m_data(Eigen::VectorXd::Zero(s)) {
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

float Vector::operator[](size_t i_index) const {
    return m_data(i_index);
}

bool Vector::setCoeff(size_t i_index, float value) {
    if(i_index >= m_size) {
        return false;
    }
    m_data(i_index) = value;
    return true;
}


// Linear algebra operations

Vector Vector::operator*(const float scalar){
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

float Vector::dot(const Vector& other){
    if(m_size != other.m_size) {
        std::cout << "ERROR: Vectors must be of the same size for dot product." << std::endl;
        return 0.0f;
    }
    float result = 0;
    for(size_t i = 0; i < m_size; ++i) {
        result += m_data(i) * other.m_data(i);
    }
    return result;    
}

