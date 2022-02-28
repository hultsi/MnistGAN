#include <iostream>
#include <string>
#include <fstream>

#include "NeuralNet.h"
#include "statpack.h"

int main(int argc, char *argv[]) {
    std::cout << "Running...\n";
    std::string images = "";
    if (argc >= 3) {
        images = argv[1];
    }
    std::vector<std::vector<float>> real { { 1,0,1,0 } };

    // Set up generator
    NeuralNet generator;
    generator.learnRate = .1;
    generator.addLayer(1);
    generator.addLayer(2);
    generator.addLayer(4);
    generator.setCostFunction("log-gdz");
    generator.setActivationFunction("sigmoid");
    generator.build();
    generator.randomizeWeightsAndBiases();
    generator.inputMin = -1;
    generator.inputMax = 1;
    generator.targetMin = 0;
    generator.targetMax = 1;

    // Set up discriminator
    NeuralNet discriminator;
    discriminator.learnRate = .1;
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

    generator.GANLink = &discriminator;

    statpack::Random::seed(123);
    int iterations = 0;
    std::ofstream genLossStream("./gloss.txt");
    std::ofstream discLossStream("./dloss.txt");
    constexpr const int BATCH_SIZE = 5;
    while (true) {
        // Run N number of epochs and check loss
        // Update both with fake data
        for (int k = 0; k < BATCH_SIZE; ++k) {
            std::vector<float> in = { statpack::Random::Float(-1.0, 1.0) };
            std::vector<float> out = generator.generate(in);
            std::vector<float> prob = discriminator.generate(out);
            discriminator.backPropagate(prob, BATCH_SIZE, false);
            generator.backPropagate(prob, BATCH_SIZE, false);
        }
        discriminator.applyDeltas();
        generator.applyDeltas();

        // Update discriminator with real data
        for (int k = 0; k < BATCH_SIZE; ++k) {
            const int ind = statpack::Random::Int(0, real.size() - 1);
            std::vector<float> prob = discriminator.generate(real[ind]);
            discriminator.backPropagate(prob, BATCH_SIZE, true);
        }
        discriminator.applyDeltas();

        // Calculate losses
        std::vector<float> in = { statpack::Random::Float(-1.0, 1.0) };
        std::vector<float> out = generator.generate(in);
        std::vector<float> prob = discriminator.generate(out);

        float loss = generator.costFunctionPointer(prob, {}, false);
        genLossStream << loss << "\n";
        loss = discriminator.costFunctionPointer(prob, {}, false);
        discLossStream << loss << "\n";

        ++iterations;
        if (iterations > 50000) {
            break;
        }
    }
    genLossStream.close();
    discLossStream.close();

    std::vector<float> tmp = generator.generate({ -.7 });
    std::cout << tmp[0] << " " << tmp[1] << " " << tmp[2] << " " << tmp[3] << "\n";
    std::vector<float> tmp2 = generator.generate({ .5 });
    std::cout << tmp2[0] << " " << tmp2[1] << " " << tmp2[2] << " " << tmp2[3] << "\n";

    std::cout << "Ending...\n";
    return 0;
}
