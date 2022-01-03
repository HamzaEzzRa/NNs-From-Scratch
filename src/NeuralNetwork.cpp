#include "NeuralNetwork.h"

#include <cassert>
#include <cstdint>

#include <random>
#include <vector>

#include "Matrix.h"
#include "Gradient.h"

std::random_device rd;
std::mt19937 gen(rd());

float NeuralNetwork::UniformDistribution_m11(const float& value)
{
    std::uniform_real_distribution<> d(-1.0, 1.0);
    return (float)d(gen);
}

float NeuralNetwork::NormalDistribution_01(const float& value)
{
    std::normal_distribution<> d{0, 1};
    return (float)d(gen);
}

NeuralNetwork::NeuralNetwork()
:learningRate(0.01f) {}

NeuralNetwork::NeuralNetwork(float learningRate)
:learningRate(learningRate) {}

void NeuralNetwork::InputLayer(uint32_t neurons)
{
    if (this->topology.size() > 0)
    {
        this->topology[0] = neurons;
    }
    else
    {
        this->topology.push_back(neurons);
    }
}

void NeuralNetwork::AddLayer(uint32_t neurons, Gradient grad)
{
    this->topology.push_back(neurons);
    this->grads.push_back(grad);
}

void NeuralNetwork::InitializeWeights(std::function<float(const float&)> initFunction)
{
    this->weights.clear();

    for (uint32_t i = 0; i < this->topology.size() - 1; i++)
    {
        Matrix layerWeights = Matrix(this->topology[i], this->topology[i + 1]);
        layerWeights = layerWeights.Apply(initFunction);
        this->weights.push_back(std::move(layerWeights));
    }
}

void NeuralNetwork::InitializeBiases(std::function<float(const float&)> initFunction)
{
    this->biases.clear();

    for (uint32_t i = 0; i < this->topology.size() - 1; i++)
    {
        Matrix layerBiases = Matrix(1, this->topology[i + 1]);
        layerBiases = layerBiases.Apply(initFunction);
        this->biases.push_back(std::move(layerBiases));
    }
}

void NeuralNetwork::FeedForward(std::vector<float> input)
{
    assert(input.size() == this->topology[0]);

    this->values.clear();

    Matrix layerValues = Matrix(1, input.size(), input);
    for (uint32_t i = 0; i < this->topology.size() - 1; i++)
    {
        Matrix tmp = layerValues.Copy();

        this->values.push_back(std::move(tmp));
        layerValues = layerValues.Dot(this->weights[i]);
        layerValues = layerValues.Add(this->biases[i]);
        layerValues = layerValues.Apply(this->grads[i].forward);
    }
    this->values.push_back(std::move(layerValues));
}

void NeuralNetwork::BackPropagate(std::vector<float> target)
{
    assert(target.size() == this->topology.back());

    Matrix layerError = Matrix(1, target.size(), target);
    layerError = layerError.Subtract(this->values.back());

    for (int32_t i = this->weights.size() - 1; i >= 0; i--)
    {
        Matrix transposedWeights = this->weights[i].Transpose();
        Matrix prevLayerError = layerError.Dot(transposedWeights);

        Matrix derivedOutput = this->values[i + 1].Apply(this->grads[i].backward);
        Matrix layerGradient = layerError.MultiplyElements(derivedOutput);
        layerGradient = layerGradient * (this->learningRate);

        Matrix transposedValues = this->values[i].Transpose();
        Matrix corrections = transposedValues.Dot(layerGradient);

        this->weights[i] = this->weights[i].Add(corrections);
        this->biases[i] = this->biases[i].Add(layerGradient);

        layerError = prevLayerError;
    }
}

std::vector<float> NeuralNetwork::GetOutput()
{
    return this->values.back().Flat();
}