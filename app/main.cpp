#include <iostream>
#include <string>
#include <fstream>
#include <algorithm>

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
    generator.learnRate = .5;
    generator.addLayer(1);
    // generator.addLayer(4);
    generator.addLayer(4);
    generator.setCostFunction("log-gdz");
    generator.setActivationFunction("relu");
    generator.build();
    generator.randomizeWeightsAndBiases();
    generator.inputMin = -1;
    generator.inputMax = 1;
    generator.targetMin = 0;
    generator.targetMax = 1;

    // Set up discriminator
    NeuralNet discriminator;
    discriminator.learnRate = .5;
    discriminator.addLayer(4);
    // discriminator.addLayer(3);
    // discriminator.addLayer(2);
    discriminator.addLayer(1);
    discriminator.setCostFunction("log-dz");
    discriminator.setActivationFunction("relu");
    discriminator.build();
    discriminator.randomizeWeightsAndBiases();
    discriminator.inputMin = generator.targetMin;
    discriminator.inputMax = generator.targetMax;
    discriminator.targetMin = 0;
    discriminator.targetMax = 1;

    generator.GANLink = &discriminator;

    statpack::Random::seed();
    int iterations = 0;
    std::ofstream genLossStream("./g_loss.txt");
    std::ofstream discLossStream("./d_loss.txt");
    std::ofstream genWeightStream("./g_w1.txt");
    std::ofstream genBiasStream("./g_b1.txt");
    std::ofstream discWeightStream("./d_w1.txt");
    std::ofstream discBiasStream("./d_b1.txt");
    constexpr const int BATCH_SIZE_1 = 5;
    constexpr const int BATCH_SIZE_2 = 5;
    while (true) {
        // Run N number of epochs and check loss
        // Update both with fake data
        for (int k = 0; k < BATCH_SIZE_1; ++k) {
            std::vector<float> in = { statpack::Random::Float(-1.0, 1.0) };
            std::vector<float> out = generator.generate(in);
            std::vector<float> prob = discriminator.generate(out);
            discriminator.backPropagate(prob, BATCH_SIZE_1, false);
            generator.backPropagate(prob, BATCH_SIZE_1, false);
        }
        discriminator.applyDeltas();
        generator.applyDeltas();

        // Update discriminator with real data
        for (int k = 0; k < BATCH_SIZE_2; ++k) {
            const int max = static_cast<int>(std::max(real.size(), static_cast<size_t>(std::numeric_limits<int>::max())));
            const int ind = statpack::Random::Int(0, max - 1);
            std::vector<float> prob = discriminator.generate(real[ind]);
            discriminator.backPropagate(prob, BATCH_SIZE_2, true);
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

        genWeightStream << generator.layers[0].weights[0][0] << "\t" <<
                           generator.layers[0].weights[1][0] << "\t" <<
                           generator.layers[0].weights[2][0] << "\t" <<
                           generator.layers[0].weights[3][0] << "\n";
        genBiasStream << generator.layers[0].biases[0] << "\t" <<
                         generator.layers[0].biases[1] << "\t" <<
                         generator.layers[0].biases[2] << "\t" <<
                         generator.layers[0].biases[3] << "\n";
        discWeightStream << discriminator.layers[0].weights[0][1] << "\t" <<
                           discriminator.layers[0].weights[0][2] << "\t" <<
                           discriminator.layers[0].weights[0][3] << "\t" <<
                           discriminator.layers[0].weights[0][4] << "\n";
        discBiasStream << discriminator.layers[0].biases[0] << "\n";

        ++iterations;
        if (iterations > 3000) {
            break;
        }
    }
    genLossStream.close();
    discLossStream.close();
    genWeightStream.close();
    genBiasStream.close();
    discWeightStream.close();
    discBiasStream.close();

    std::vector<float> tmp = generator.generate({ -.7 });
    std::cout << tmp[0] << " " << tmp[1] << " " << tmp[2] << " " << tmp[3] << "\n";
    std::vector<float> tmp2 = generator.generate({ .5 });
    std::cout << tmp2[0] << " " << tmp2[1] << " " << tmp2[2] << " " << tmp2[3] << "\n";

    std::cout << "Ending...\n";
    return 0;
}
