#include <iostream>

#include <vector>

#include <cstdint>

#include "Gradient.h"
#include "NeuralNetwork.h"

int main(int argc, char** argv)
{
    NeuralNetwork net = NeuralNetwork(0.1f);

    std::vector<std::vector<float>> input = {
        {0.0f, 0.0f},
        {1.0f, 0.0f},
        {0.0f, 1.0f},
        {1.0f, 1.0f}
    };
    std::vector<std::vector<float>> targetOutput = {
        {0.0f},
        {1.0f},
        {1.0f},
        {0.0f}
    };

    net.InputLayer(2);
    net.AddLayer(4, Gradient::ReLU);
    net.AddLayer(1, Gradient::ReLU);

    net.InitializeWeights(NeuralNetwork::NormalDistribution_01);
    net.InitializeBiases([](const float& value)
    {
        return 0.0f;
    });

    uint32_t epochs = 2500;
    for (uint32_t i = 0; i < epochs; i++)
    {
        int index = rand() % input.size();
        net.FeedForward(input[index]);
        net.BackPropagate(targetOutput[index]);
    }

    for (auto in : input)
    {
        net.FeedForward(in);
        auto prediction = net.GetOutput();
        std::cout << in[0] << " XOR " << in[1] << " ----> " << prediction[0] << std::endl;
    }

    return 0;
}