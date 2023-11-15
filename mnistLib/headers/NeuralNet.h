#pragma once

#include <vector>
#include <cstddef>
#include <functional>
#include <cassert>
#include <iostream>

#include "statpack.h"
#include "mnistParser.h"
#include "templates.h"

class NeuralNet {
public:
    float learnRate;
    float inputMin;
    float inputMax;
    float targetMin;
    float targetMax;
    float activationMin;
    float activationMax;
    
    // Generator part of GAN needs knowledge of the first layer of the
    // GAN's discriminator. Set this to point to the discriminator of
    // the GAN in the generator.
    const NeuralNet *GANLink = nullptr;

    std::ofstream outLossStream;

    std::function<float(const std::vector<float>&, const std::vector<float>&, bool)> costFunctionPointer;
    std::function<float(float, float, bool)> dCostFunctionPointer;

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
    
    std::vector<Layer> layers;

    NeuralNet();
    void addLayer(size_t size);
    void build();
    void randomizeWeightsAndBiases(unsigned int seed = 0);
    float train(const std::vector<float> &inputs, const std::vector<float> &target, const float epoch = 1.f, const bool realData = true);
    // TODO: add batch standardization
    std::vector<float> generate(const std::vector<float> &inputs);
    void forwardPropagate(const std::vector<float> &inputs);
    void backPropagate(const std::vector<float>& target, const float batchSize = 1.f, const bool realData = true);
    void applyDeltas();
    void setCostFunction(std::string name);
    void setActivationFunction(std::string name);

private:
    struct CostFunctions {
        /**
         *  The derivatives here are for a _single index_
         *  and _not_ over a whole set of points (since that is what we need)
         *  
         *  Usage of [[maybe_unused]] is to be able to use single function
         *  pointer for every cost function easily.
         */
        static float mse(const std::vector<float> &predicted, const std::vector<float> &observed = {}, [[maybe_unused]] const bool realData = true) {
            #ifdef CUSTOM_DEBUG
                assert(!(observed.size() != predicted.size()) && "Vector sizes are not equal.");
            #endif
            float out = 0;
            for (size_t i = 0; i < observed.size(); ++i) {
                out += std::pow(observed[i] - predicted[i], 2.f);
            }
            // TODO: Will precision here ever be an issue?
            return out / static_cast<float>(predicted.size());
        }

        static float dMse(float predicted, float observed, [[maybe_unused]] const bool realData = true) {
            return 2 * (observed - predicted);
        }

        static float logDz(const std::vector<float> &predicted, [[maybe_unused]] const std::vector<float> &observed = {},  const bool realData = true) {
            float out = 0;
            if (realData) {
                for (size_t i = 0; i < predicted.size(); ++i) {
                    out += (predicted[i] <= 0 ? -templates::logn(std::numeric_limits<float>::min()) : -std::log(predicted[i]));
                }
            } else {
                for (size_t i = 0; i < predicted.size(); ++i) {
                    out += (predicted[i] >= 1 ? -templates::logn(1 - std::numeric_limits<float>::min()) : -std::log(1 - predicted[i]));
                }
            }
            // TODO: Will precision here ever be an issue?
            return out / static_cast<float>(predicted.size());
        }

        static float dLogDz(float predicted, [[maybe_unused]] float observed = 0, const bool realData = true) {
            if (realData) {
                if (predicted <= .0001) {
                    return -9999;
                }
                return -1 / predicted;
            } else {
                if (predicted >= .9999) {
                    return 9999;
                }
                return 1 / (1 - predicted);
            }
        }

        static float logGdz(const std::vector<float> &predicted, [[maybe_unused]] const std::vector<float> &observed = {}, [[maybe_unused]] const bool realData = true) {
            float out = 0;
            for (size_t i = 0; i < predicted.size(); ++i) {
                out += (predicted[i] <= 0 ? -std::log(std::numeric_limits<float>::min()) : -std::log(predicted[i]));
            }
            // TODO: Will precision here ever be an issue?
            return out / static_cast<float>(predicted.size());
        }

        static float dLogGdz(float predicted, [[maybe_unused]] float observed = 0, [[maybe_unused]] const bool realData = true) {
            return predicted - 1;
        }
    };

    struct ActivationFunctions {
        static float sigmoid(float x) {
            return 1.0f / (1.0f + std::exp(-x));
        }

        static float dSigmoid(float x) {
            const float s = sigmoid(x);
            return s * (1.0f - s);
        }

        static float relu(float x) {
            return std::max(0.0f, x);
        }

        static float dRelu(float x) {
            return x > 0 ? 1 : 0;
        }
    };

    float costFunction(const std::vector<float> &predicted, const std::vector<float> &observed = {}, const bool realData = true);
    float dCostFunction(float predicted,  float observed = 0., const bool realData = true);
};