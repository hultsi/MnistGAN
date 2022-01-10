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
    std::vector<float> real { 2,3,5,7,11,13,17,19,23,29,31,37,41,43,47};

    // Set up generator
    NeuralNet generator;
    generator.learnRate = .1;
    generator.addLayer(1);
    generator.addLayer(4);
    generator.addLayer(1);
    generator.setCostFunction("log-gdz");
    generator.setActivationFunction("sigmoid");
    generator.build();
    generator.randomizeWeightsAndBiases();
    generator.inputMin = -1;
    generator.inputMax = 1;
    generator.targetMin = 0;
    generator.targetMax = 100;

    // Set up discriminator
    NeuralNet discriminator;
    discriminator.learnRate = .1;
    discriminator.addLayer(1);
    discriminator.addLayer(4);
    discriminator.addLayer(4);
    discriminator.addLayer(1);
    discriminator.setCostFunction("log-dz");
    discriminator.setActivationFunction("sigmoid");
    discriminator.build();
    discriminator.randomizeWeightsAndBiases();
    discriminator.inputMin = generator.targetMin;
    discriminator.inputMax = generator.targetMax;
    discriminator.targetMin = 0;
    discriminator.targetMax = 1;

    int iterations = 0;
    while (true) {
        // Run N number of epochs and check loss
        // Update both with fake data
        std::vector<float> in = { statpack::Random::Float(-1, 1) };
        std::vector<float> out = generator.generate(in);
        std::vector<float> prob = discriminator.generate(out);
        // std::cout << "FAKE PRIME PROBABILITY: " << prob[0] << "\n";
        discriminator.backPropagate(prob, 1, false);
        discriminator.applyDeltas();
        generator.backPropagate(prob, 1, false);
        generator.applyDeltas();
        
        // Update discriminator with real data
        prob = discriminator.generate({ real[statpack::Random::Int(0, real.size() - 1)] });
        // std::cout << "REAL PRIME PROBABILITY: " << prob[0] << "\n";
        discriminator.backPropagate(prob, 1, true);
        discriminator.applyDeltas();

        ++iterations;
        if (iterations > 500) {
            break;
        }
    }

    std::cout << generator.generate({ statpack::Random::Float(-1, 1) })[0] << "\n";
    std::cout << generator.generate({ statpack::Random::Float(-1, 1) })[0] << "\n";
    std::cout << generator.generate({ statpack::Random::Float(-1, 1) })[0] << "\n";
    std::cout << generator.generate({ statpack::Random::Float(-1, 1) })[0] << "\n";
    std::cout << generator.generate({ statpack::Random::Float(-1, 1) })[0] << "\n";
    std::cout << generator.generate({ statpack::Random::Float(-1, 1) })[0] << "\n";

    std::cout << "Ending...\n";
    return 0;
}
