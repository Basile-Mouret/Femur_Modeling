== Optimizing Memory Allocations


The main bottleneck of our Neural Network implementation is memory bandwidth as most of the training and inference time is lost waiting for data to be loaded from memory.
This meant we had to optimize the memory allocations of our computations.

In order to increase our performance, we adopted a data oriented design programming approach by increasing memory cache efficiency. As matrices are the largest data structures of our programs, we had to understand how they were stored.
There are two main convention, in `C` multidimensional arrays are row-major, so they are stored from left to right then up to down.
On the contrary, in `Fortran` arrays are stored in columns-major order.

#figure(
  image("../resources/img/Row_and_column_major_order.svg", width: 20%),
  caption: [Illustration of row- and column-major order by CMG Lee.]
)

For compatibility reasons with linear algebra libraries in `Fortran` like `BLAS` and `LAPACK`, `Eigen` uses the column-major order by default.
This means elements from a same columns are near one another in memory.
So when we access elements of a matrix, we should iterate over elements from the same columns as they will already be loaded in the cache, resulting in faster load times.


#figure(
  grid(
    columns: (auto, auto, auto),
    align: horizon,
    ```cpp 
    for (size_t j=0; j < m_rows; ++j){
      for (size_t i=0; i < m_cols; ++i){
        // operation
      }
    }
    ```,
    $quad --> quad$,
    ```cpp 
    for (size_t j=0; j < m_cols; ++j){
      for (size_t i=0; i < m_rows; ++i){
        // operation
      }
    }
    ```,
  ),
  caption: [Exchanging iteration order results in 2x faster \ model training by reducing cache misses]
)

We then reduced the amount of memory that was created and destroyed during the neural network methods by preallocating matrices and vectors.
These are placeholders in memory created before a function call and that are reused, thus removing the overhead of allocating and destroying memory. The allocation and destruction overhead.

Finally we combined some linear algebra operations to remove temporary matrices. For example when computing $y = A^t dot x$ during the backpropagation, we were creating a temporary transpose matrix :  

```cpp
Matrix2D<T> W_transpose = m_weights[layer + 1].transpose();
Vector<T> weightedDelta = W_transpose * deltas[lastLayer - layer - 1];
```
In order to eliminate this, we created a new method on the `Matrix2D` class that combines both operations efficiently :

#block(breakable: false)[
```cpp
Vector<T> multiplyTranspose(const Vector<T>& vec) const {
    Vector<T> result(m_cols);
    
    for (size_t j = 0; j < m_cols; ++j) {
        T dot = 0;
        for (size_t i = 0; i < m_rows; ++i) {
            dot += m_data(i, j) * vec(i);
        }
        result(j) = dot;
    }
    return result;
}
```]

// describe and show usage of transposeMultiply and Rank1Update

In total, optimizing our memory usage made our training epochs eight times faster.
// we are still at 7s per epoch wich is quite long, we could still improve our code




