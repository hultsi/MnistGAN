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
    // std::vector<std::vector<float>> real { 
    //     { 3 },
    //     { 5 },
    //     { 7 },
    //     { 9 },
    //     { 11 }
    // };
    std::vector<std::vector<float>> real { { 1,0,1,0 } };

    // Set up generator
    NeuralNet generator;
    generator.learnRate = 0.01;
    generator.addLayer(1);
    generator.addLayer(4);
    // generator.addLayer(1);
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
    discriminator.learnRate = 0.01;
    // discriminator.addLayer(1);
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

    statpack::Random::seed(123);
    int iterations = 0;
    std::ofstream tmp1("./gloss.txt");
    std::ofstream tmp2("./dloss.txt");
    while (true) {
        // Run N number of epochs and check loss
        // Update both with fake data
        std::vector<float> in = { statpack::Random::Float(-1.0, 1.0) };
        std::vector<float> out = generator.generate(in);
        std::vector<float> prob = discriminator.generate(out);

        float loss = generator.costFunctionPointer(prob, {}, false);
        tmp1 << loss << "\n";
        loss = discriminator.costFunctionPointer(prob, {}, false);
        tmp2 << loss << "\n";

        discriminator.backPropagate(prob, 1, false);
        generator.backPropagate(discriminator.layers[0].weights[0], prob[0], 1, false);
        discriminator.applyDeltas();
        generator.applyDeltas();
        
        // Update discriminator with real data
        prob = discriminator.generate(real[statpack::Random::Int(0, real.size() - 1)]);

        discriminator.backPropagate(prob, 1, true);
        discriminator.applyDeltas();

        ++iterations;
        if (iterations > 10000) {
            break;
        }
    }
    tmp1.close();
    tmp2.close();

    std::vector<float> tmp = generator.generate({ -.7 });
    std::cout << tmp[0] << " " << tmp[1] << " " << tmp[2] << " " << tmp[3] << "\n";
    // std::vector<float> tmp2 = generator.generate({ 0 });
    // std::cout << tmp2[0] << "\n";
    // std::vector<float> tmp3 = generator.generate({ .9 });
    // std::cout << tmp3[0] << "\n";
    // std::vector<float> tmp4 = generator.generate({ .2 });
    // std::cout << tmp4[0] << "\n";

    std::cout << "Ending...\n";
    return 0;
}
