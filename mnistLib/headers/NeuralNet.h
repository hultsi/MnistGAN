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
    float learnRate;
    float inputMin;
    float inputMax;
    float targetMin;
    float targetMax;
    float activationMin;
    float activationMax;

    std::function<float(const std::vector<float>&, const std::vector<float>&)> costFunction;
    std::function<float(float, float)> dCostFunction;

    std::function<float(float)> activationFunction;
    std::function<float(float)> dActivationFunction;

    std::vector<float> targetVector;

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
        learnRate(0.01f),
        inputMin(0.0f),
        inputMax(1.0f),
        targetMin(0.0f),
        targetMax(1.0f),
        costFunction(CostFunctions::mse),
        dCostFunction(CostFunctions::dMse),
        activationFunction(ActivationFunctions::sigmoid),
        dActivationFunction(ActivationFunctions::dSigmoid)
    {}

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
        for (size_t i = 0; i < layers.size() - 1; ++i) {
            layers[i].delta_nodes.resize(layers[i].sizeIn);
        }
        for (size_t i = 1; i < layers.size(); ++i) {
            layers[i].wSum.resize(layers[i].sizeIn);
        }
        targetVector.resize(layers[layers.size() - 1].sizeIn);
    }

    void randomizeWeightsAndBiases(unsigned int seed = 0) {
        statpack::Random::seed(seed);

        for (size_t i = 0; i < layers.size() - 1; ++i) {
            for (size_t m = 0; m < layers[i].biases.size(); ++m) {
                layers[i].biases[m] = statpack::Random::Float(-3, 3);
            }
            for (size_t m = 0; m < layers[i].weights.size(); ++m) {
                for (size_t k = 0; k < layers[i].weights[m].size(); ++k) {
                    layers[i].weights[m][k] = statpack::Random::Float(-3, -3);
                }
            }
        }
    }

    float train(const std::vector<float> &inputs, const std::vector<float> &target, const float epoch = 1) {
#ifdef CUSTOM_DEBUG
        assert(inputs.size() == layers[0].weights[0].size() && "Input vector has an incorrect size.");
        assert(target.size() == layers[layers.size()-1].nodes.size() && "Target and ouput vectors have different lengths.");
#endif
        for (size_t i = 0; i < inputs.size(); ++i) {
            layers[0].nodes[i] = statpack::normalize(inputs[i], inputMin, inputMax, -1.0f, 1.0f); // -1 to 1 works fine
        }
        for (size_t i = 0; i < target.size(); ++i) {
            targetVector[i] = statpack::normalize(target[i], targetMin, targetMax, activationMin, activationMax); // output limits depends on the activation function
        }
        
        forwardPropagate(inputs);
        backPropagate(targetVector, epoch);
        return costFunction(targetVector, layers[layers.size() - 1].nodes);
    }

    std::vector<float> generate(const std::vector<float> &inputs) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            layers[0].nodes[i] = statpack::normalize(inputs[i], inputMin, inputMax, -1.0f, 1.0f); // -1 to 1 works fine
        }
        forwardPropagate(inputs);

        std::vector<float> out;
        out.resize(layers[layers.size() - 1].nodes.size());
        for (size_t i = 0; i < layers[layers.size() - 1].nodes.size(); ++i) {
            out[i] = statpack::normalize(layers[layers.size() - 1].nodes[i], activationMin, activationMax, targetMin, targetMax);
        }
        return out;
    }
    
    void applyDeltas() {
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

    void setCostFunction(std::string name) {
        if (name == "mse") {
            costFunction = CostFunctions::mse;
            dCostFunction = CostFunctions::dMse;
        }
    }

    void setActivationFunction(std::string name) {
        if (name == "sigmoid") {
            activationFunction = ActivationFunctions::sigmoid;
            dActivationFunction = ActivationFunctions::dSigmoid;
            activationMin = 0.0f;
            activationMax = 1.0f;
        }
    }

private:
    std::vector<Layer> layers;

    struct CostFunctions {
        /**
         *  The derivatives here are for a _single index_
         *  and _not_ over a whole set of points (since that is what we need)
         */
        static float mse(const std::vector<float> &observed, const std::vector<float> &predicted) {
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

        static float logDz(const std::vector<float> &predicted, const bool realData = true) {
            float mse = 0;
            if (realData) {
                for (size_t i = 0; i < predicted.size(); ++i) {
                    mse += (predicted[i] <= 0 ? -std::log(std::numeric_limits<float>::min()) : -std::log(predicted[i]));
                }
            } else {
                for (size_t i = 0; i < predicted.size(); ++i) {
                    mse += (predicted[i] >= 1 ? -std::log(std::numeric_limits<float>::min()) : -std::log(1 - predicted[i]));
                }
            }
            return mse / predicted.size();
        }

        static float dLogDz(float predicted, const bool realData = true) {
            if (realData) {
                return (predicted <= 0 ? -std::log(std::numeric_limits<float>::min()) : -(1 - std::log(predicted)));
            } else {
                return predicted;
            }
        }

        // static float logGdz(const std::vector<float> &predicted) {
        //     float mse = 0;
        //     for (size_t i = 0; i < predicted.size(); ++i) {
        //         mse += (predicted[i] <= 0 ? -std::log(std::numeric_limits<float>::min()) : -std::log(predicted[i]));
        //     }
        //     return mse / predicted.size();
        // }

        // static float dLogGdz(float predicted) {
        //     return (predicted <= 0 ? -std::log(std::numeric_limits<float>::min()) : -(1 - std::log(predicted)));
        // }
    };

    struct ActivationFunctions {
        static float sigmoid(float x) {
            return 1.0f / (1.0f + std::exp(-x));
        }

        static float dSigmoid(float x) {
            const float s = sigmoid(x);
            return s * (1.0f - s);
        }
    };

    void forwardPropagate(const std::vector<float> &inputs) {
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

    void backPropagate(const std::vector<float>& target, const float epoch) {
        // First layer calculation differs slightly from the rest
        const size_t lastLayer = layers.size() - 1;
        for (size_t k = 0; k < layers[lastLayer].sizeIn; ++k) {
            const float bpTerm = dActivationFunction(layers[lastLayer].wSum[k]) * dCostFunction(layers[lastLayer].nodes[k], target[k]);
            for (size_t n = 0; n < layers[lastLayer - 1].sizeIn; ++n) {
                layers[lastLayer - 1].delta_weights[k][n] += layers[lastLayer - 1].nodes[n] * bpTerm / epoch;
                layers[lastLayer - 1].delta_nodes[n] += layers[lastLayer - 1].weights[k][n] * bpTerm / layers[lastLayer].sizeIn;
            }
            layers[lastLayer - 1].delta_biases[k] += bpTerm / epoch;
        }

        if (lastLayer - 1 == 0) return;
        
        for (size_t i = layers.size() - 2; i > 0; --i) {
            for (size_t k = 0; k < layers[i].sizeIn; ++k) {
                const float bpTerm = dActivationFunction(layers[i].wSum[k]) * layers[i].delta_nodes[k];
                for (size_t n = 0; n < layers[i - 1].sizeIn; ++n) {
                    layers[i - 1].delta_weights[k][n] += layers[i - 1].nodes[n] * bpTerm / epoch;
                    layers[i - 1].delta_nodes[n] += layers[i - 1].weights[k][n] * bpTerm / layers[i].sizeIn;
                }
                layers[i - 1].delta_biases[k] += bpTerm / epoch;
                layers[i].delta_nodes[k] = 0;
            }
        }
    }
};