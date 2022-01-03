#include "Gradient.h"

#include <functional>

#include <cmath>

Gradient Gradient::Sigmoid = Gradient(
    Gradient::SgmdFunc,
    Gradient::DrvSgmdFunc
);

Gradient Gradient::ReLU = Gradient(
    Gradient::ReLUFunc,
    Gradient::DrvReLUFunc
);

Gradient::Gradient(std::function<float(const float&)> forward,
    std::function<float(const float&)> backward)
:forward(forward),
backward(backward) {}

float Gradient::SgmdFunc(const float& value)
{
    return 1.0f / (1 + exp(-value));
}

float Gradient::DrvSgmdFunc(const float& value)
{
    return value * (1 - value);
}

float Gradient::ReLUFunc(const float& value)
{
    return (value <= 0) ? 0 : value;
}

float Gradient::DrvReLUFunc(const float& value)
{
    return (value <= 0) ? 0 : 1;
}