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
    NeuralNet nn;
    nn.addLayer(4);
    nn.addLayer(2);
    nn.build();
    nn.randomizeWeightsAndBiases();
    // nn.train();
    std::cout << "Ending...\n";
    return 0;
}
