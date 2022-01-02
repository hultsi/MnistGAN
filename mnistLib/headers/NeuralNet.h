#pragma once

#include <fstream>
#include <vector>
#include <cstddef>
#include <cassert>

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
    std::vector<Layer> layers;
    
    NeuralNet() {};

    void addLayer(size_t size) {
        layers.emplace_back(Layer(size));
    }

    void build() {
#ifdef CUSTOM_DEBUG
        assert(layers.size() >= 2 && "NeuralNet requires at least 2 layers (input & output) to work)");
#endif
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            layers.at(i).biases.resize(layers[i+1].sizeIn);
            layers.at(i).delta_biases.resize(layers[i+1].sizeIn);
            layers.at(i).weights.resize(layers.at(i).sizeIn);
            layers.at(i).delta_weights.resize(layers.at(i).sizeIn);
            layers.at(i).sizeOut = layers[i + 1].sizeIn;
            for (size_t k = 0; k < layers.at(i).sizeIn; ++k) {
                layers.at(i).weights[k].resize(layers.at(i).sizeOut);
                layers.at(i).delta_weights[k].resize(layers.at(i).sizeOut);
            }
        }
        for (size_t i = 1; i < layers.size() - 1; ++i) {
            layers.at(i).delta_nodes.resize(layers.at(i).sizeIn);
        }
    }

    void train() {

    }
};