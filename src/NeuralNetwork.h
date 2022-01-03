#ifndef __NEURAL_NETWORK_H__
#define __NEURAL_NETWORK_H__

#include <cstdint>
#include <functional>

#include <vector>

#include "Matrix.h"
#include "Gradient.h"

class NeuralNetwork
{
public:
    float learningRate;
    std::vector<Matrix> biases;
    std::vector<Matrix> weights;
    std::vector<Gradient> grads;

public:
    static float UniformDistribution_m11(const float& value);
    static float NormalDistribution_01(const float& value);

public:
    NeuralNetwork();
    NeuralNetwork(float learningRate);
    void InputLayer(uint32_t neurons);
    void AddLayer(uint32_t neurons, Gradient grad);
    
    void InitializeWeights(std::function<float(const float&)> initFunc);
    void InitializeBiases(std::function<float(const float&)> initFunc);
    
    void FeedForward(std::vector<float> input);
    void BackPropagate(std::vector<float> target);
    
    std::vector<float> GetOutput();

private:
    std::vector<uint32_t> topology;
    std::vector<Matrix> values;
};

#endif