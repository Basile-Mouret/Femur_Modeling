#include "linalg.hpp"
#include <iostream>
#include <vector>

int main(){

    Matrix2D<float> A(3, 2);
    Matrix2D<float> B(2, 3, 1);
    Matrix2D<float> C(B);

    std::cout << "Matrix A (3x2 zero matrix):\n" << A << std::endl;
    std::cout << "Matrix B (2x3 initialized to 1):\n" << B << std::endl;
    std::cout << "Matrix C (copy of B):\n" << C << std::endl;
    std::cout << "A == B? " << (A == B ? "True" : "False") << std::endl;
    std::cout << "B == C? " << (B == C ? "True" : "False") << std::endl;
    std::cout << "Is A zero? " << (A.isZero() ? "True" : "False") << std::endl;

    Matrix2DSquare<float> D(2);
    Matrix2DSquare<float> E(2, 5);
    Matrix2DSquare<float> F(E);

    std::cout << "Matrix D (2x2 zero square matrix):\n" << D << std::endl;
    std::cout << "Matrix E (2x2 square matrix initialized to 5):\n" << E << std::endl;
    std::cout << "Matrix F (copy of E):\n" << F << std::endl;


    std::cout << "A : \n" << A << " \n Number of rows: " << A.getSizeRows() << "\n Number of cols: " << A.getSizeCols() << std::endl;

    A.setCoeff(0, 0, 3.5);
    std::cout << "After setting A(0,0) to 3.5:\n" << A << std::endl;
    Vector<float> row0 = A.getRow(0);
    std::cout << "Row 0 of A: " << row0 << std::endl;
    Vector<float> col1 = A.getCol(1);
    std::cout << "Column 1 of A: " << col1 << std::endl;

    Matrix2D<float> G(2, 2);
    G.setCoeff(0, 0, 1);
    G.setCoeff(0, 1, 2);
    G.setCoeff(1, 0, 3);
    G.setCoeff(1, 1, 4);
    std::cout << "Matrix G (2x2 initialized):\n" << G << std::endl;
    Matrix2D<float> H = G * 2.0;
    std::cout << "Matrix H (G multiplied by 2):\n" << H << std::endl;

    Matrix2D<float> I = H + G;
    std::cout << "Matrix I (H + G):\n" << I << std::endl;
    Matrix2D<float> J = I - G;
    std::cout << "Matrix J (I - G):\n" << J << std::endl;
    Matrix2D<float> K = G * J;
    std::cout << "Matrix K (G * J):\n" << K << std::endl;

    Vector<float> v(2);
    v.setCoeff(0, 1);
    v.setCoeff(1, 2);
    std::cout << "Vector v:\n" << v << std::endl;
    Vector<float> w = G * v;
    std::cout << "Vector w (G * v):\n" << w << std::endl;

    return 0;
}