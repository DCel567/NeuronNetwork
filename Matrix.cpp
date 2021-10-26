#include "Matrix.h"

Matrix::Matrix(int cols, int rows){
    this->m(cols, std::vector<double>(rows, 0));
}