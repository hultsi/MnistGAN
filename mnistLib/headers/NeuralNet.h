#pragma once

#include <fstream>
#include <vector>
#include <cstddef>
#include <cassert>
#include <iostream>

#include "statpack.h"
#include "mnistParser.h"

class NeuralNet {
public:
    struct Layer {
        size_t sizeIn;
        size_t sizeOut;
        std::vector<float> nodes;
        std::vector<float> delta_nodes;
        std::vector<std::vector<float>> weights;
        std::vector<std::vector<float>> delta_weights;
        std::vector<float> biases;
        std::vector<float> delta_biases;

        Layer(size_t size) : sizeIn(size) { 
            nodes.resize(size);
        };

        float forward(int nodeIndex) {
            return statpack::weightedSum(nodes, weights[nodeIndex]) + biases[nodeIndex];
        }

        void backward() {

        }
    };
    
    NeuralNet() {};

    void addLayer(size_t size) {
        layers.emplace_back(Layer(size));
    }

    void build() {
#ifdef CUSTOM_DEBUG
        assert(layers.size() >= 2 && "NeuralNet requires at least 2 layers (input & output) to work)");
#endif
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            // Biases
            layers[i].biases.resize(layers[i+1].sizeIn);
            layers[i].delta_biases.resize(layers[i+1].sizeIn);
            // Weights
            layers[i].sizeOut = layers[i + 1].sizeIn;
            layers[i].weights.resize(layers[i].sizeOut);
            layers[i].delta_weights.resize(layers[i].sizeOut);
            for (size_t k = 0; k < layers[i].sizeOut; ++k) {
                layers[i].weights[k].resize(layers[i].sizeIn);
                layers[i].delta_weights[k].resize(layers[i].sizeIn);
            }
        }
        for (size_t i = 1; i < layers.size() - 1; ++i) {
            layers[i].delta_nodes.resize(layers[i].sizeIn);
        }
    }

    void randomizeWeightsAndBiases() {
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            for (size_t m = 0; m < layers[i].biases.size(); ++m) {
                layers[i].biases[m] = statpack::Random::Float(-1, 1);
            }
            for (size_t m = 0; m < layers[i].weights.size(); ++m) {
                for (size_t k = 0; k < layers[i].weights[m].size(); ++k) {
                    layers[i].weights[m][k] = statpack::Random::Float(-1, -1);
                }
            }
        }
    }

    void train() {
        // Forward propagation
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            for (size_t k = 0; k < layers[i].sizeOut; ++k) {
                const float wSum = layers[i].forward(k);
                layers[i+1].nodes[k] = statpack::sigmoid(wSum);
            }
        }

        // Backward propagation

    }

private:
    std::vector<Layer> layers;
};