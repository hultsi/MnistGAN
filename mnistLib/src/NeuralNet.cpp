#include <vector>
#include <cstddef>
#include <functional>
#include <cassert>
#include <iostream>

#include "NeuralNet.h"

NeuralNet::NeuralNet() : 
        learnRate(0.01f),
        inputMin(0.0f),
        inputMax(1.0f),
        targetMin(0.0f),
        targetMax(1.0f),
        costFunctionPointer(CostFunctions::mse),
        dCostFunctionPointer(CostFunctions::dMse),
        activationFunction(ActivationFunctions::sigmoid),
        dActivationFunction(ActivationFunctions::dSigmoid)
    {}

void NeuralNet::addLayer(size_t size) {
    layers.emplace_back(Layer(size));
}

void NeuralNet::build() {
#ifdef CUSTOM_DEBUG
    assert(layers.size() >= 2 && "NeuralNet requires at least 2 layers (input & output) to work)");
#endif
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        // Biases
        layers[i].biases.resize(layers[i + 1].sizeIn);
        layers[i].delta_biases.resize(layers[i + 1].sizeIn);
        // Weights
        layers[i].sizeOut = layers[i + 1].sizeIn;
        layers[i].weights.resize(layers[i].sizeOut);
        layers[i].delta_weights.resize(layers[i].sizeOut);
        for (size_t k = 0; k < layers[i].sizeOut; ++k) {
            layers[i].weights[k].resize(layers[i].sizeIn);
            layers[i].delta_weights[k].resize(layers[i].sizeIn);
        }
    }
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        layers[i].delta_nodes.resize(layers[i].sizeIn);
    }
    for (size_t i = 1; i < layers.size(); ++i) {
        layers[i].wSum.resize(layers[i].sizeIn);
    }
    targetVector.resize(layers[layers.size() - 1].sizeIn);
}

void NeuralNet::randomizeWeightsAndBiases(unsigned int seed) {
    statpack::Random::seed(seed);
    for (auto& layer : layers) {
        for (auto& bias : layer.biases) {
            bias = statpack::Random::Float(-10, 10);
        }
        for (auto& weightArr : layer.weights) {
            for (auto& weight : weightArr) {
                weight = statpack::Random::Float(-10, 10);
            }
        }
    }
}

float NeuralNet::train(const std::vector<float> &inputs, const std::vector<float> &target, const float epoch, const bool realData) {
#ifdef CUSTOM_DEBUG
    assert(inputs.size() == layers[0].weights[0].size() && "Input vector has an incorrect size.");
    assert(target.size() == layers[layers.size()-1].nodes.size() && "Target and ouput vectors have different lengths.");
#endif
    for (size_t i = 0; i < inputs.size(); ++i) {
        layers[0].nodes[i] = inputs[i]; // statpack::normalize(inputs[i], inputMin, inputMax, -1.0f, 1.0f); // -1 to 1 works fine
    }
    for (size_t i = 0; i < target.size(); ++i) {
        targetVector[i] = statpack::normalize(target[i], targetMin, targetMax, activationMin, activationMax); // output limits depends on the activation function
    }
    
    forwardPropagate(inputs);
    backPropagate(targetVector, epoch);
    return costFunction(layers[layers.size() - 1].nodes, targetVector, realData);
}

// TODO: add batch standardization
std::vector<float> NeuralNet::generate(const std::vector<float> &inputs) {
    for (size_t i = 0; i < inputs.size(); ++i) {
        layers[0].nodes[i] = inputs[i]; // statpack::normalize(inputs[i], inputMin, inputMax, -1.0f, 1.0f); // -1 to 1 works fine
    }
    forwardPropagate(layers[0].nodes);

    std::vector<float> out;
    out.resize(layers[layers.size() - 1].nodes.size());
    for (size_t i = 0; i < layers[layers.size() - 1].nodes.size(); ++i) {
        out[i] = statpack::normalize(layers[layers.size() - 1].nodes[i], activationMin, activationMax, targetMin, targetMax);
    }
    return out;
}

void NeuralNet::forwardPropagate(const std::vector<float> &inputs) {
    // First layer calculation differs slightly from the rest
    for (size_t k = 0; k < layers[0].sizeOut; ++k) {
        layers[1].wSum[k] = statpack::weightedSum(inputs, layers[0].weights[k]) + layers[0].biases[k];
        layers[1].nodes[k] = activationFunction(layers[1].wSum[k]);
    }

    for (size_t i = 1; i < layers.size() - 1; ++i) {
        for (size_t k = 0; k < layers[i].sizeOut; ++k) {
            layers[i+1].wSum[k] = statpack::weightedSum(layers[i].nodes, layers[i].weights[k]) + layers[i].biases[k];
            layers[i+1].nodes[k] = activationFunction(layers[i+1].wSum[k]);
        }
    }
}

void NeuralNet::backPropagate(const std::vector<float>& target, const float batchSize, const bool realData) {
    // First layer calculation differs slightly from the rest
    const size_t lastLayer = layers.size() - 1;
    for (size_t k = 0; k < layers[lastLayer].sizeIn; ++k) {
        float bpTerm = 0;
        if (GANLink) {
            bpTerm = GANLink->layers[0].weights[0][k] * dActivationFunction(layers[lastLayer].wSum[k]) * dCostFunction(target[0], layers[lastLayer].nodes[k], realData);
        } else {
            bpTerm = dActivationFunction(layers[lastLayer].wSum[k]) * dCostFunction(target[k], layers[lastLayer].nodes[k], realData);
        }
        for (size_t n = 0; n < layers[lastLayer - 1].sizeIn; ++n) {
            layers[lastLayer - 1].delta_weights[k][n] += layers[lastLayer - 1].nodes[n] * bpTerm / batchSize;
            layers[lastLayer - 1].delta_nodes[n] += layers[lastLayer - 1].weights[k][n] * bpTerm / static_cast<float>(layers[lastLayer].sizeIn);
        }
        layers[lastLayer - 1].delta_biases[k] += bpTerm / batchSize;
    }

    if (lastLayer == 1) return;
    
    for (size_t i = layers.size() - 2; i > 0; --i) {
        for (size_t k = 0; k < layers[i].sizeIn; ++k) {
            const float bpTerm = dActivationFunction(layers[i].wSum[k]) * layers[i].delta_nodes[k];
            for (size_t n = 0; n < layers[i - 1].sizeIn; ++n) {
                layers[i - 1].delta_weights[k][n] += layers[i - 1].nodes[n] * bpTerm / batchSize;
                layers[i - 1].delta_nodes[n] += layers[i - 1].weights[k][n] * bpTerm / static_cast<float>(layers[i].sizeIn);
            }
            layers[i - 1].delta_biases[k] += bpTerm / batchSize;
            layers[i].delta_nodes[k] = 0;
        }
    }
}

void NeuralNet::applyDeltas() {
    for (size_t i = 0; i < layers.size() - 1; ++i) {
        for (size_t k = 0; k < layers[i].sizeOut; ++k) {
            for (size_t n = 0; n < layers[i].sizeIn; ++n) {
                layers[i].weights[k][n] -= layers[i].delta_weights[k][n] * learnRate; 
                layers[i].delta_weights[k][n] = 0;
            }
            layers[i].biases[k] -= layers[i].delta_biases[k] * learnRate;
            layers[i].delta_biases[k] = 0;
        }
    }
}

void NeuralNet::setCostFunction(std::string name) {
    if (name == "mse") {
        costFunctionPointer = CostFunctions::mse;
        dCostFunctionPointer = CostFunctions::dMse;
    } else if (name == "log-dz") {
        costFunctionPointer = CostFunctions::logDz;
        dCostFunctionPointer = CostFunctions::dLogDz;
    } else if (name == "log-gdz") {
        costFunctionPointer = CostFunctions::logGdz;
        dCostFunctionPointer = CostFunctions::dLogGdz;
    }
}

void NeuralNet::setActivationFunction(std::string name) {
    if (name == "sigmoid") {
        activationFunction = ActivationFunctions::sigmoid;
        dActivationFunction = ActivationFunctions::dSigmoid;
        activationMin = 0.0f;
        activationMax = 1.0f;
    } else if (name == "relu") {
        activationFunction = ActivationFunctions::relu;
        dActivationFunction = ActivationFunctions::dRelu;
        activationMin = 0.0f;
        activationMax = 1.0f;
    }
}

float NeuralNet::costFunction(const std::vector<float> &predicted, const std::vector<float> &observed, const bool realData) {
    return costFunctionPointer(predicted, observed, realData);
}

float NeuralNet::dCostFunction(float predicted,  float observed, const bool realData) {
    return dCostFunctionPointer(predicted, observed, realData);
}