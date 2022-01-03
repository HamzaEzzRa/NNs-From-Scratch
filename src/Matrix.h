#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <iostream>

#include <cstdint>
#include <functional>
#include <vector>

class Matrix
{
public:
    uint32_t nRows;
    uint32_t nCols;

    Matrix(uint32_t nRows, uint32_t nCols);
    Matrix(uint32_t nRows, uint32_t nCols, float value);
    Matrix(uint32_t nRows, uint32_t nCols, std::vector<float> values);

    Matrix Copy();

    Matrix Apply(std::function<float(const float&)> function);

    float& At(uint32_t row, uint32_t col);
    Matrix Negative();
    Matrix Transpose();

    Matrix Add(Matrix& other);
    Matrix Subtract(Matrix& other);
    Matrix Dot(Matrix& other);
    Matrix MultiplyElements(Matrix& other);

    Matrix operator+ (float scalar);
    Matrix operator- (float scalar);
    Matrix operator* (float scalar);
    Matrix operator/ (float scalar);

    std::vector<float>& Flat();
    
    friend std::ostream& operator<< (std::ostream& os, Matrix& mat);

private:
    std::vector<float> values;
};

#endif