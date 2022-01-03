#ifndef __GRADIENT_H__
#define __GRADIENT_H__

#include <functional>

class Gradient
{
public:
    static Gradient Sigmoid;
    static Gradient ReLU;

public:
    std::function<float(const float&)> forward;
    std::function<float(const float&)> backward;
    
    Gradient(std::function<float(const float&)> forward, std::function<float(const float&)> backward);

private:
    static float SgmdFunc(const float& value);
    static float DrvSgmdFunc(const float& value);

    static float ReLUFunc(const float& value);
    static float DrvReLUFunc(const float& value);
};

#endif