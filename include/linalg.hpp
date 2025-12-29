/**
 * @file linalg.hpp
 * @brief Linear algebra library providing Vector and Matrix classes
 * @details This file contains template classes for linear algebra operations including
 *          vectors, general 2D matrices, and square matrices. Uses Eigen library for
 *          efficient computation.
 */

#ifndef LINALG_HPP
#define LINALG_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>

// Forward declarations
template<typename T>
class Matrix2D;

/**
 * @class Vector
 * @brief Template class representing a mathematical vector
 * @tparam T Data type of vector elements (e.g., int, float, double)
 * 
 * This class provides a mathematical vector implementation with various
 * linear algebra operations including addition, subtraction, scalar and
 * dot products, and Hadamard products.
 */
template<typename T>
class Vector{
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const Vector<U>& vec);

    protected:
        size_t m_size;                                      ///< Size (dimension) of the vector
        Eigen::Matrix<T, Eigen::Dynamic, 1> m_data;         ///< Internal Eigen column vector storage

    public:
        // Constructors
        
        /**
         * @brief Constructs a zero vector of given size
         * 
         * Creates a vector of specified size initialized to zero.
         * 
         * @param s Size of the vector
         */
        Vector(size_t s);

        /**
         * @brief Constructs a vector of given size with uniform initialization
         * 
         * Creates a vector where all elements are set to the specified initial value.
         * 
         * @param s Size of the vector
         * @param init_value Value to initialize all elements
         */
        Vector(size_t s, T init_value);

        /**
         * @brief Constructs a vector from a std::vector
         * 
         * Creates a vector of specified size and initializes it with values
         * from a standard vector.
         * 
         * @param s Size of the vector
         * @param init_values Standard vector containing initialization values
         */
        Vector(size_t s, const std::vector<T>& init_values);

        /**
         * @brief Copy constructor
         * 
         * Creates a deep copy of another vector.
         * 
         * @param other Vector to copy
         */
        Vector(const Vector<T>& other);

        // Basic operations

        /**
         * @brief Gets the size (dimension) of the vector
         * 
         * @return Size of the vector
         */
        size_t getSize() const;

        /**
         * @brief Checks if the vector is a zero vector
         * 
         * Determines if all elements of the vector are zero.
         * 
         * @return true if all elements are zero, false otherwise
         */
        bool isZero() const;

        /**
         * @brief Equality comparison operator
         * 
         * Compares this vector with another for equality.
         * 
         * @param other Vector to compare with
         * @return true if vectors are equal, false otherwise
         */
        bool operator==(const Vector<T>& other) const;

        /**
         * @brief Subscript operator for read-only access
         * 
         * Accesses an element at the specified index (const version).
         * 
         * @param i_index Index of the element to access
         * @return Value at the specified index
         */
        T operator()(size_t i_index) const;

        /**
         * @brief Subscript operator for read-write access
         * 
         * Accesses an element at the specified index (non-const version).
         * 
         * @param i_index Index of the element to access
         * @return Reference to the element at the specified index
         */
        T &operator()(size_t i_index);

        /**
         * @brief Sets the coefficient at a specific index
         * 
         * Modifies the value of the vector element at the given index.
         * 
         * @param i_index Index where to set the value
         * @param value New value to set
         * @return true if successful, false if index is out of bounds
         */
        bool setCoeff(size_t i_index, T value);



        // Linear algebra operations

        /**
         * @brief Scalar multiplication operator
         * 
         * Multiplies each element of the vector by a scalar value.
         * 
         * @param scalar Scalar value to multiply by
         * @return New vector containing the result
         */
        Vector<T> operator*(const T scalar);

        /**
         * @brief Vector addition operator
         * 
         * Adds two vectors element-wise. If sizes do not match, no operation
         * is performed.
         * 
         * @param other Vector to add
         * @return New vector containing the sum
         */
        Vector<T> operator+(const Vector<T> &other);

        /**
         * @brief Vector subtraction operator
         * 
         * Subtracts another vector element-wise. If sizes do not match,
         * no operation is performed.
         * 
         * @param other Vector to subtract
         * @return New vector containing the difference
         */
        Vector<T> operator-(const Vector<T> &other);


        /**
         * @brief Computes the dot product with another vector
         * 
         * Calculates the scalar product (inner product) of two vectors.
         * Returns 0 if sizes do not match.
         * 
         * @param other Vector to compute dot product with
         * @return Scalar dot product value
         */
        T dot(const Vector<T>& other);

        /**
         * @brief Computes the Hadamard product (element-wise multiplication)
         * 
         * Performs element-wise multiplication with another vector.
         * Returns a zero vector if sizes do not match.
         * 
         * @param other Vector to multiply element-wise
         * @return New vector containing the element-wise product
         */
        Vector<T> hadamard(const Vector<T>& other) const;

        /**
         * @brief Computes the outer product with another vector
         * 
         * Calculates the outer product resulting in a matrix of size
         * (this.size x other.size).
         * 
         * @param other Vector to compute outer product with
         * @return Matrix2D<T> containing the outer product result
         */
        Matrix2D<T> outerProduct(const Vector<T>& other) const;
};


/**
 * @class Matrix2D
 * @brief Template class representing a 2D matrix
 * @tparam T Data type of matrix elements (e.g., int, float, double)
 * 
 * This class provides a mathematical 2D matrix implementation with various
 * linear algebra operations including addition, subtraction, multiplication,
 * transposition, and matrix-vector operations.
 */
template<typename T>
class Matrix2D{
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const Matrix2D<U>& mat);
    
    protected:
        size_t m_rows;                                                      ///< Number of rows in the matrix
        size_t m_cols;                                                      ///< Number of columns in the matrix
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_data;           ///< Internal Eigen matrix storage

    public:
        // Constructors

        /**
         * @brief Constructs a zero matrix of given dimensions
         * 
         * Creates a matrix of specified size initialized to zero.
         * 
         * @param rows Number of rows
         * @param cols Number of columns
         */
        Matrix2D(size_t rows, size_t cols);

        /**
         * @brief Constructs a matrix with uniform initialization
         * 
         * Creates a matrix where all elements are set to the specified initial value.
         * 
         * @param rows Number of rows
         * @param cols Number of columns
         * @param init_value Value to initialize all elements
         */
        Matrix2D(size_t rows, size_t cols, const T init_value);

        /**
         * @brief Copy constructor
         * 
         * Creates a deep copy of another matrix.
         * 
         * @param other Matrix to copy
         */
        Matrix2D(const Matrix2D<T>& other);



        // Basic operations

        /**
         * @brief Gets the number of rows in the matrix
         * 
         * @return Number of rows
         */
        size_t getSizeRows() const;

        /**
         * @brief Gets the number of columns in the matrix
         * 
         * @return Number of columns
         */
        size_t getSizeCols() const;

        /**
         * @brief Extracts a row as a vector
         * 
         * Returns the specified row as a Vector object. Prints an error message
         * and returns a zero vector if the row index is out of bounds.
         * 
         * @param i_row Row index (0-based)
         * @return Vector<T> containing the row elements
         */
        Vector<T> getRow(size_t i_row) const;

        /**
         * @brief Extracts a column as a vector
         * 
         * Returns the specified column as a Vector object. Prints an error message
         * and returns a zero vector if the column index is out of bounds.
         * 
         * @param i_col Column index (0-based)
         * @return Vector<T> containing the column elements
         */
        Vector<T> getCol(size_t i_col) const;

        /**
         * @brief Sets a matrix coefficient at a specific position
         * 
         * Modifies the value at the specified row and column.
         * 
         * @param i_row Row index (0-based)
         * @param i_col Column index (0-based)
         * @param value New value to set
         * @return true if successful, false if indices are out of bounds
         */
        bool setCoeff(size_t i_row, size_t i_col, T value);

        /**
         * @brief Sets an entire row from a vector
         * 
         * Replaces the specified row with values from a vector.
         * 
         * @param i_row Row index (0-based)
         * @param row Vector containing new row values
         * @return true if successful, false if sizes do not match
         */
        bool setRow(size_t i_row, const Vector<T>& row);

        /**
         * @brief Sets an entire column from a vector
         * 
         * Replaces the specified column with values from a vector.
         * 
         * @param i_col Column index (0-based)
         * @param col Vector containing new column values
         * @return true if successful, false if sizes do not match
         */
        bool setCol(size_t i_col, const Vector<T>& col);

        /**
         * @brief Checks if the matrix is a zero matrix
         * 
         * Determines if all elements of the matrix are zero.
         * 
         * @return true if all elements are zero, false otherwise
         */
        bool isZero() const;

        /**
         * @brief Equality comparison operator
         * 
         * Compares this matrix with another for equality.
         * 
         * @param other Matrix to compare with
         * @return true if matrices are equal, false otherwise
         */       
        bool operator==(const Matrix2D& other) const;

        /**
         * @brief Accesses an element (const version)
         * 
         * Returns the value at the specified row and column.
         * 
         * @param i_row Row index (0-based)
         * @param i_col Column index (0-based)
         * @return Value at the specified position
         */
       T operator()(size_t i_row, size_t i_col) const;

        /**
         * @brief Accesses an element (non-const version)
         * 
         * Returns a reference to the element at the specified position.
         * 
         * @param i_row Row index (0-based)
         * @param i_col Column index (0-based)
         * @return Reference to the element at the specified position
         */
       T &operator()(size_t i_row, size_t i_col);

        // Linear algebra operations

        /**
         * @brief Matrix scalar multiplication operator
         * 
         * Multiplies each element of the matrix by a scalar value.
         * 
         * @param scalar Scalar value to multiply by
         * @return New matrix containing the result
         */
        Matrix2D operator*(const float scalar);

        /**
         * @brief Matrix addition operator
         * 
         * Adds two matrices element-wise. If sizes do not match, returns
         * the original matrix without modification.
         * 
         * @param other Matrix to add
         * @return New matrix containing the sum
         */
        Matrix2D operator+(const  Matrix2D &other);

        /**
         * @brief Matrix subtraction operator
         * 
         * Subtracts another matrix element-wise. If sizes do not match,
         * returns the original matrix without modification.
         * 
         * @param other Matrix to subtract
         * @return New matrix containing the difference
         */
        Matrix2D operator-(const  Matrix2D &other);

        /**
         * @brief Matrix multiplication operator
         * 
         * Performs standard matrix multiplication. If dimensions are not
         * compatible, returns the original matrix without modification.
         * 
         * @param other Matrix to multiply with
         * @return New matrix containing the product
         */
        Matrix2D operator*(const Matrix2D &other);

        /**
         * @brief Matrix-vector multiplication operator
         * 
         * Multiplies the matrix by a vector. Returns a zero vector if
         * dimensions are not compatible.
         * 
         * @param vec Vector to multiply with
         * @return Resulting vector from multiplication
         */
        Vector<T> operator*(const Vector<T> &vec);

        /**
         * @brief Computes the transpose of the matrix
         * 
         * Returns a new matrix with rows and columns swapped.
         * 
         * @return Transposed matrix
         */
        Matrix2D<T> transpose() const;
};

/**
 * @class Matrix2DSquare
 * @brief Template class representing a square matrix
 * @tparam T Data type of matrix elements (e.g., int, float, double)
 * 
 * This class inherits from Matrix2D and represents a special case where
 * the number of rows equals the number of columns. Useful for operations
 * that require square matrices (e.g., determinants, eigenvalues).
 */
template<typename T>
class Matrix2DSquare : public Matrix2D<T> {
    public:
        /**
         * @brief Constructs a zero square matrix of given size
         * 
         * Creates a square matrix of specified size initialized to zero.
         * 
         * @param size Number of rows and columns
         */
        Matrix2DSquare(size_t size);

        /**
         * @brief Constructs a square matrix with uniform initialization
         * 
         * Creates a square matrix where all elements are set to the
         * specified initial value.
         * 
         * @param size Number of rows and columns
         * @param init_value Value to initialize all elements
         */
        Matrix2DSquare(size_t size, const T init_value);

        /**
         * @brief Copy constructor
         * 
         * Creates a deep copy of another square matrix.
         * 
         * @param other Square matrix to copy
         */
        Matrix2DSquare(const Matrix2DSquare<T>& other);
};

#endif
