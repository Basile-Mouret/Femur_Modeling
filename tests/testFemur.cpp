#include "Femur.hpp"


int main() {
    Femur femur("../data/L_Femur_11_DECIM.obj.FINAL.obj");
    Femur femur2(femur);
    Femur femur3;
    femur2.saveToFile("../data/testsave.obj");
    return 0;
}
