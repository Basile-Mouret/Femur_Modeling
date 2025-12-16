#ifndef LINALG_HPP
#define LINALG_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>

class Vector{
    friend std::ostream& operator<<(std::ostream& os, const Vector& vec);

    protected:
        size_t m_size;
        Eigen::VectorXd m_data;

    public:
        //Constructors
        Vector(size_t s);

        Vector(size_t s, float init_value);

        Vector(size_t s, const std::vector<float>& init_values);

        Vector(const Vector& other);

        // Basic operations

        /*
        * @brief Get the Size object
        * @return size_t
        */
        size_t getSize() const;

        /*
        * @brief Check if the vector is a zero vector
        * @return bool
        */
        bool isZero() const;

        /*
        * @brief Equality operator overload
        * @param other: const Vector&
        * @return bool
        */
        bool operator==(const Vector& other) const;

        /*
        * @brief Subscript operator overload for read-only access
        * @param index: size_t
        * @return float
        */
        float operator[](size_t i_index) const;

        /*
        * @brief Set the coefficient of the vector at a specific index
        * @param index: size_t
        * @param value: float
        * @return bool
        */
        bool setCoeff(size_t i_index, float value);



        // Linear algebra operations

        /*
        * @brief Scalar multiplication. Returns a reference to a new vector after multiplication.
        * @param scalar: float
        * @return Vector&
        */
        Vector operator*(const float scalar);

        /*
        * @brief Vector addition. Returns a reference to a new vector after addition. If sizes do not match, no operation is performed.
        * @param other: const Vector&
        * @return Vector&
        */
        Vector operator+(const Vector &other);

        /*
        * @brief Vector subtraction. Returns a reference to a new vector after subtraction. If sizes do not match, no operation is performed.
        * @param other: const Vector&
        * @return Vector&
        */
        Vector operator-(const Vector &other);


        /*
        * @brief Compute the dot product with another vector. Return 0 if sizes do not match.
        * @param other: const Vector&
        * @return float
        */
        float dot(const Vector& other);
};


#endif