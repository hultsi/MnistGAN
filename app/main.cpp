#include <iostream>
#include <string>
#include "statpack.h"
#include "NeuralNet.h"

int main(int argc, char *argv[]) {
    std::cout << "Running...\n";
    std::string images = "";
    if (argc >= 3) {
        images = argv[1];
    }
    // mnistGAN::NeuralNet<2,4,6> nn;
    // std::cout << nn.LAYERS << "\n";
    // std::cout << std::get<0>(nn.layers).size() << "\n";
    // std::cout << std::get<1>(nn.layers).size() << "\n";
    // std::cout << std::get<0>(nn.weights).size() << "\n";
    mnistGAN::NeuralNet<1,22,13> nn;
    std::cout << nn.LAYERS << "\n";
    std::cout << nn.nodes.size() << "\n";
    std::cout << nn.test << "\n"; 
    std::cout << "Ending...\n";
    return 0;
}
