#include <iostream>
#include <string>
#include "NeuralNet.h"
#include "statpack.h"

int main(int argc, char *argv[]) {
    std::cout << "Running...\n";
    std::string images = "";
    if (argc >= 3) {
        images = argv[1];
    }
    std::vector<float> real { 2,3,5,7,11,13,17,19,23,29,31,37 };

    // Set up generator
    NeuralNet generator;
    generator.learnRate = .1;
    generator.addLayer(2);
    generator.addLayer(4);
    generator.addLayer(1);
    generator.setCostFunction("mse");
    generator.setActivationFunction("sigmoid");
    generator.build();
    generator.randomizeWeightsAndBiases();
    generator.inputMin = -1;
    generator.inputMax = 1;
    generator.targetMin = 2;
    generator.targetMax = 37;

    // Set up discriminator
    NeuralNet discriminator;
    discriminator.learnRate = .1;
    discriminator.addLayer(1);
    discriminator.addLayer(4);
    discriminator.addLayer(4);
    discriminator.addLayer(1);
    discriminator.setCostFunction("mse");
    discriminator.setActivationFunction("sigmoid");
    discriminator.build();
    discriminator.randomizeWeightsAndBiases();
    discriminator.inputMin = generator.targetMin;
    discriminator.inputMax = generator.targetMax;
    discriminator.targetMin = 0;
    discriminator.targetMax = 1;

    constexpr size_t epoch = 3;
    while (true) {
        // Run N number of epochs and check loss
        constexpr size_t iterations = 1;
        // float loss = 0;
        for (size_t k = 0; k < iterations; ++k) {
            // Run over one epoch set
            for (size_t i = 0; i < epoch; ++i) {
                std::vector<float> in = { statpack::Random::Float(-1, 1), statpack::Random::Float(-1, 1) };
                std::vector<float> out = generator.generate(in);
                std::vector<float> prob = discriminator.generate(out);
                
            }
            // nn.applyDeltas();
        }
        // std::cout << "Loss: " << loss << "\n";
        // todo: Do stuff
        // if (loss / (N * epoch) < .00001) {
        break;
        // }
    }

    std::cout << "Ending...\n";
    return 0;
}
