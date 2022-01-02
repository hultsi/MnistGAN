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

        void forward() {
            for (int i = 0; i < sizeOut; ++i) {
                // wSum1[i] = statpack::weightedSum<inputLayerSize>(inputs, weights1[i]) + bias1[i];
                // hiddenNeuron1[i] = statpack::sigmoid(wSum1[i]);
            }
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
        assert(layers.size() >= 2 && "NeuralNet requires at least 2 layers (input & output) to work)");
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            layers[i].biases.resize(layers[i+1].sizeIn);
            layers[i].delta_biases.resize(layers[i+1].sizeIn);
            layers[i].weights.resize(layers[i].sizeIn);
            layers[i].delta_weights.resize(layers[i].sizeIn);
            layers[i].sizeOut = layers[i + 1].sizeIn;
            for (size_t k = 0; k < layers[i].sizeIn; ++k) {
                layers[i].weights[k].resize(layers[i].sizeOut);
                layers[i].delta_weights[k].resize(layers[i].sizeOut);
            }
        }
        for (size_t i = 1; i < layers.size() - 1; ++i) {
            layers[i].delta_nodes.resize(layers[i].sizeIn);
        }
    }

};