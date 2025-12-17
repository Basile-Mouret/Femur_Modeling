#ifndef LINALG_HPP
#define LINALG_HPP

#include <iostream>
#include <vector>
#include <Eigen/Dense>

template<typename T>
class Vector{
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const Vector<U>& vec);

    protected:
        size_t m_size;
        Eigen::Matrix<T, Eigen::Dynamic, 1> m_data;

    public:
        //Constructors
        /*
        * @brief Construct a Vector null of given size
        * @param s: size_t
        */
        Vector(size_t s);

        /*
        * @brief Construct a Vector of given size initialized to init_value
        * @param s: size_t
        * @param init_value: T
        */
        Vector(size_t s, T init_value);

        /*
        * @brief Construct a Vector of given size initialized to init_values
        * @param s: size_t
        * @param init_values: const std::vector<T>&
        */
        Vector(size_t s, const std::vector<T>& init_values);

        /*
        * @brief Copy constructor
        * @param other: const Vector<T>&
        * @return Vector<T>
        */
        Vector(const Vector<T>& other);

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
        * @param other: const Vector<T>&
        * @return bool
        */
        bool operator==(const Vector<T>& other) const;

        /*
        * @brief Subscript operator overload for read-only access
        * @param index: size_t
        * @return T
        */
        T operator()(size_t i_index) const;

        /*
        * @brief Set the coefficient of the vector at a specific index
        * @param index: size_t
        * @param value: T
        * @return bool
        */
        bool setCoeff(size_t i_index, T value);



        // Linear algebra operations

        /*
        * @brief Scalar multiplication. Returns a reference to a new vector after multiplication.
        * @param scalar: T
        * @return Vector<T>
        */
        Vector<T> operator*(const T scalar);

        /*
        * @brief Vector addition. Returns a reference to a new vector after addition. If sizes do not match, no operation is performed.
        * @param other: const Vector<T>&
        * @return Vector<T>
        */
        Vector<T> operator+(const Vector<T> &other);

        /*
        * @brief Vector subtraction. Returns a reference to a new vector after subtraction. If sizes do not match, no operation is performed.
        * @param other: const Vector<T>&
        * @return Vector<T>
        */
        Vector<T> operator-(const Vector<T> &other);


        /*
        * @brief Compute the dot product with another vector. Return 0 if sizes do not match.
        * @param other: const Vector<T>&
        * @return T
        */
        T dot(const Vector<T>& other);
};


template<typename T>
class Matrix2D{
    template<typename U>
    friend std::ostream& operator<<(std::ostream& os, const Matrix2D<U>& mat);
    
    protected:
        size_t m_rows;
        size_t m_cols;
        Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic> m_data;

    public:
        // Constructors

        /*
        * @brief Construct a Matrix2D of given sizes initialized to zero
        * @param rows: size_t
        * @param cols: size_t
        */
        Matrix2D(size_t rows, size_t cols);

        /*
        * @brief Construct a Matrix2D of given sizes initialized to init_value
        * @param rows: size_t
        * @param cols: size_t
        * @param init_value: T
        */
        Matrix2D(size_t rows, size_t cols, const T init_value);

        /*
        * @brief Copy constructor
        * @param other: const Matrix2D<T>&
        */
        Matrix2D(const Matrix2D<T>& other);



        // Basic operations

        /*
        * @brief Get the number of rows
        * @return size_t
        */
        size_t getSizeRows() const;

        /*
        * @brief Get the number of columns
        * @return size_t
        */
        size_t getSizeCols() const;

        /*
        * @brief Get the Row in a Matrix2D as a Vector. Print an error message if i_row is out of bounds and return a zero Vector.
        * @param i_row: size_t
        * @return Vector<T>
        */
        Vector<T> getRow(size_t i_row) const;

        /*
        * @brief Get the Column in a Matrix2D as a Vector. Print an error message if i_col is out of bounds and return a zero Vector.
        * @param i_col: size_t
        * @return Vector<T>
        */
        Vector<T> getCol(size_t i_col) const;

        /*
        * @brief Set the coefficient of the matrix at a specific row and column. Return false if indices are out of bounds.
        * @param i_row: size_t
        * @param i_col: size_t
        * @param value: T
        * @return bool
        */
        bool setCoeff(size_t i_row, size_t i_col, T value);

        /*
        *@brief Set the Row in a Matrix2D from a Vector. Return false if sizes do not match.
        *@param i_row: size_t
        *@param row: const Vector<T>&
        *@return bool
        */
        bool setRow(size_t i_row, const Vector<T>& row);

        /*
        *@brief Set the Column in a Matrix2D from a Vector. Return false if sizes do not match.
        *@param i_col: size_t
        *@param col: const Vector<T>&
        *@return bool
        */
        bool setCol(size_t i_col, const Vector<T>& col);

        /*
        * @brief Check if the matrix is a zero matrix
        * @return bool
        */
        bool isZero() const;

        /*
        * @brief Equality operator overload
        * @param other: const Matrix2D&
        * @return bool
        */       
        bool operator==(const Matrix2D& other) const;

        /*
        * @brief Access an element in the matrix
        * @param i_row: size_t
        * @param i_col: size_t
        * @return T
        */
       T operator()(size_t i_row, size_t i_col) const;


        // Linear algebra operations

        /*
        * Matrix scalar multiplication. Returns a new matrix after multiplication.
        * @param scalar: float
        * @return Matrix2D&
        */
        Matrix2D operator*(const float scalar);

        /*
        * @brief Matrix addition. Returns a new matrix after addition. If sizes do not match, no operation is performed. Return the original matrix.
        * @param other: const Matrix2D&
        * @return Matrix2D
        */
        Matrix2D operator+(const  Matrix2D &other);

        /*
        * @brief Matrix subtraction. Returns a new matrix after subtraction. If sizes do not match, no operation is performed. Return the original matrix.
        * @param other: const Matrix2D&
        * @return Matrix2D
        */
        Matrix2D operator-(const  Matrix2D &other);

        /*
        * @brief Matrix multiplication. Returns a new matrix after multiplication. If sizes are not compatible, no operation is performed. Return the original matrix.
        * @param other: const Matrix2D&
        * @return Matrix2D
        */
        Matrix2D operator*(const Matrix2D &other);

        /*
        * @brief Matrix-Vector multiplication. Returns a new vector after multiplication. If sizes are not compatible, return a zero vector.
        * @param vec: const Vector&
        * @return Vector
        */
        Vector<T> operator*(const Vector<T> &vec);
};

template<typename T>
class Matrix2DSquare : public Matrix2D<T> {
    public:
        /*
        * @brief Construct a Square Matrix2D of given size initialized to zero
        * @param size: size_t
        */
        Matrix2DSquare(size_t size);

        /*
        * @brief Construct a Square Matrix2D of given size initialized to init_value
        * @param size: size_t
        * @param init_value: T
        */
        Matrix2DSquare(size_t size, const T init_value);

        /*
        * @brief Copy constructor
        * @param other: const Matrix2DSquare<T>&
        */
        Matrix2DSquare(const Matrix2DSquare<T>& other);
};

#endif