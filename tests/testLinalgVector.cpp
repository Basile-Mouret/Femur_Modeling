#include "linalg.hpp"
#include <iostream>
#include <vector>

int main(){
    Vector v1(3);
    Vector v2(3, 2.0f);
    Vector v3(3, std::vector<float>{1.0f, 2.0f, 3.0f});
    Vector v4(v1);

    std::cout << "v1: " << v1 << std::endl;
    std::cout << "v2: " << v2 << std::endl;
    std::cout << "v3: " << v3 << std::endl;
    std::cout << "v4: " << v4 << std::endl;

    std::cout << "v1 size: " << v1.getSize() << std::endl;

    std::cout << "v1 is zero: " << (v1.isZero() ? "true" : "false") << std::endl;
    std::cout << "v2 is zero: " << (v2.isZero() ? "true" : "false") << std::endl;

    std::cout << "v1 == v4: " << (v1 == v4 ? "true" : "false") << std::endl;
    std::cout << "v1 == v2: " << (v1 == v2 ? "true" : "false") << std::endl;

    std::cout << "v3[1]: " << v3(1) << std::endl;
    v3.setCoeff(1, 5.0f);
    std::cout << "v3 after setCoeff(1, 5.0): " << v3 << std::endl;

    std::cout << "v2 * 3.0 = " << v2 << " * 3.0 = " << (v2 * 3.0f) << std::endl;
    std::cout << "v2 + v3 = " << v2 << " + " << v3 << " = " << (v2 + v3) << std::endl;
    std::cout << "v3 - v2 = " << v3 << " - " << v2 << " = " << (v3 - v2) << std::endl;
    std::cout << "v2 . v3 = " << v2 << " . " << v3 << " = " << v2.dot(v3) << std::endl;
    return 0;
}