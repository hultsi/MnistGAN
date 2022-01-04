#pragma once

#include <fstream>
#include <vector>
#include <cstddef>
#include <functional>
#include <cassert>
#include <iostream>

#include "statpack.h"
#include "mnistParser.h"

class NeuralNet {
public:
    float loss;
    size_t epochLength;

    std::function<float(std::vector<float>, std::vector<float>)> costFunction;
    std::function<float(float, float)> dCostFunction;

    struct Layer {
        size_t sizeIn;
        size_t sizeOut;
        std::vector<float> nodes;
        std::vector<float> delta_nodes;
        std::vector<std::vector<float>> weights;
        std::vector<std::vector<float>> delta_weights;
        std::vector<float> biases;
        std::vector<float> delta_biases;
        std::vector<float> wSum;

        Layer(size_t size) : sizeIn(size) { 
            nodes.resize(size);
        };
    };
    

    NeuralNet() : 
        loss(0), 
        costFunction(CostFunctions::mse),
        dCostFunction(CostFunctions::dMse)
    {}

    void addLayer(size_t size) {
        layers.emplace_back(Layer(size));
    }

    void build() {
#ifdef CUSTOM_DEBUG
        assert(layers.size() >= 2 && "NeuralNet requires at least 2 layers (input & output) to work)");
#endif
        if (!costFunction || dCostFunction) {
            costFunction = std::bind(statpack::mse<float>, std::placeholders::_1, std::placeholders::_2);
            dCostFunction = [](float observed, float predicted) -> float {
                return std::pow(observed - predicted, 2);
            };
        }

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
            if (i > 0) {
                layers[i].wSum.resize(layers[i].sizeIn);
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

    void train(std::vector<float> target) {
#ifdef CUSTOM_DEBUG
        assert(target.size() == layers[layers.size()-1].nodes.size() && "Target and ouput vectors have different lengths.");
#endif
        // Forward propagation...
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            for (size_t k = 0; k < layers[i].sizeOut; ++k) {
                layers[i+1].wSum[i] = statpack::weightedSum(layers[i].nodes, layers[i].weights[k]) + layers[i].biases[k];
                layers[i+1].nodes[k] = statpack::sigmoid(layers[i+1].wSum[i]);
            }
        }

        loss = costFunction(target, layers[layers.size()- 1].nodes);

        // Back propagation...
        // First layer calculation differs from the rest
        for (size_t i = layers.size() - 1; i > 0; --i) {
            for (size_t k = 0; k < layers[i].sizeIn; ++k) {
                const float bpTerm = statpack::dSigmoid(layers[i].wSum[k]) * dCostFunction(layers[i].nodes[k], target[k]);
                // TOdo: continue...
            }
        }
    }

    void setCostFunction(std::string name) {
        if (name == "mse") {
            costFunction = CostFunctions::mse;
            dCostFunction = CostFunctions::dMse;
        }
    }

private:
    std::vector<Layer> layers;

    struct CostFunctions {
        /**
         *  The derivatives here are for a _single index_
         *  and _not_ over a whole set of points (since that is what we need)
         */

        static float mse(std::vector<float> observed, std::vector<float> predicted) {
            #ifdef CUSTOM_DEBUG
                assert(!(observed.size() != predicted.size()) && "Vector sizes are not equal.");
            #endif
            float mse = 0;
            for (size_t i = 0; i < observed.size(); ++i) {
                mse += std::pow(observed[i] - predicted[i], 2);
            }
            return mse / observed.size();
        }

        static float dMse(float observed, float predicted) {
            return 2 * (observed - predicted);
        }
    };
};