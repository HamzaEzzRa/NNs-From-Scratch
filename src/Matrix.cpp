#include "Matrix.h"

#include <iostream>
#include <vector>

#include <cstdint>
#include <cassert>

std::ostream& operator<< (std::ostream& os, Matrix& mat)
{
    os << "[";
    for (uint32_t i = 0; i < mat.nRows; i++)
    {
        for (uint32_t j = 0; j < mat.nCols; j++)
        {
            os << mat.At(i, j);
            if (j < mat.nCols - 1)
            {
                os << " ";
            }
        }
        if (i < mat.nRows - 1)
        {
            os << std::endl;
        }
    }
    os << "]";
    return os;
}

Matrix::Matrix(uint32_t nRows, uint32_t nCols)
:nRows(nRows),
nCols(nCols)
{
    this->values.resize(nRows * nCols, 0.0f);
}

Matrix::Matrix(uint32_t nRows, uint32_t nCols, float value)
:nRows(nRows),
nCols(nCols)
{
    for (uint32_t i = 0; i < nRows; i++)
    {
        for (uint32_t j = 0; j < nCols; j++)
        {
            this->values.push_back(value);
        }
    }    
}

Matrix::Matrix(uint32_t nRows, uint32_t nCols, std::vector<float> values)
:nRows(nRows),
nCols(nCols)
{
    assert(nRows * nCols == values.size());

    for (uint32_t i = 0; i < nRows; i++)
    {
        for (uint32_t j = 0; j < nCols; j++)
        {
            this->values.push_back(values[i * nCols + j]);
        }
    }
}

Matrix Matrix::Copy()
{
    return Matrix(this->nRows, this->nCols, this->values);
}

Matrix Matrix::Apply(std::function<float(const float&)> function)
{
    Matrix out = Matrix(this->nRows, this->nCols);
    for (uint32_t i = 0; i < this->nRows; i++)
    {
        for (uint32_t j = 0; j < this->nCols; j++)
        {
            out.At(i, j) = function(this->At(i, j));
        }
    }
    return out;
}

float& Matrix::At(uint32_t row, uint32_t col)
{
    assert(row < nRows && col < nCols);
    return values[row * nCols + col];
}

Matrix Matrix::Negative()
{
    Matrix out = Matrix(this->nRows, this->nCols);
    for (uint32_t i = 0; i < this->nRows; i++)
    {
        for (uint32_t j = 0; j < this->nCols; j++)
        {
            out.At(i, j) = -this->At(i, j);
        }
    }
    return out;
}

Matrix Matrix::Transpose()
{
    Matrix out = Matrix(this->nCols, this->nRows);
    for (uint32_t i = 0; i < this->nRows; i++)
    {
        for (uint32_t j = 0; j < this->nCols; j++)
        {
            out.At(j, i) = this->At(i, j);
        }
    }
    return out;
}

Matrix Matrix::Add(Matrix& other)
{
    assert(this->nRows == other.nRows && this->nCols == other.nCols);

    Matrix out = Matrix(this->nRows, this->nCols);
    for (uint32_t i = 0; i < this->nRows; i++)
    {
        for (uint32_t j = 0; j < this->nCols; j++)
        {
            out.At(i, j) = this->At(i, j) + other.At(i, j);
        }
    }
    return out;
}

Matrix Matrix::Subtract(Matrix& other)
{
    Matrix negative = other.Negative();
    return Matrix::Add(negative);
}

Matrix Matrix::MultiplyElements(Matrix& other)
{
    assert(this->nRows == other.nRows && this->nCols == other.nCols);

    Matrix out = Matrix(this->nRows, this->nCols);
    for (uint32_t i = 0; i < this->nRows; i++)
    {
        for (uint32_t j = 0; j < this->nCols; j++)
        {
            out.At(i, j) = this->At(i, j) * other.At(i, j);
        }
    }
    return out;
}

Matrix Matrix::Dot(Matrix& other)
{
    assert(this->nCols == other.nRows);

    Matrix out = Matrix(this->nRows, other.nCols);
    for (uint32_t i = 0; i < out.nRows; i++)
    {
        for (uint32_t j = 0; j < out.nCols; j++)
        {
            float elem = 0.0f;
            for (uint32_t k = 0; k < this->nCols; k++)
            {
                elem += this->At(i, k) * other.At(k, j);
            }
            out.At(i, j) = elem;
        }
    }
    return out;
}

Matrix Matrix::operator+ (float scalar)
{
    Matrix out = Matrix(this->nRows, this->nCols);
    for (uint32_t i = 0; i < out.nRows; i++)
    {
        for (uint32_t j = 0; j < out.nCols; j++)
        {
            out.At(i, j) = this->At(i, j) + scalar;
        }
    }
    return out;
}

Matrix Matrix::operator- (float scalar)
{
    return *this + (-scalar);
}

Matrix Matrix::operator* (float scalar)
{
    Matrix out = Matrix(this->nRows, this->nCols);
    for (uint32_t i = 0; i < out.nRows; i++)
    {
        for (uint32_t j = 0; j < out.nCols; j++)
        {
            out.At(i, j) = this->At(i, j) * scalar;
        }
    }
    return out;
}

Matrix Matrix::operator/ (float scalar)
{
    return *this * (1.0f / scalar);
}

std::vector<float>& Matrix::Flat()
{
    return this->values;
}