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
    mnistGAN::NeuralNet<2,4,6,123> nn;
    std::cout << nn.LAYERS << "\n";
    std::cout << std::get<0>(nn.layer).size() << "\n";
    std::cout << std::get<1>(nn.layer).size() << "\n";
    std::cout << std::get<2>(nn.layer).size() << "\n";
    std::cout << std::get<3>(nn.layer).size() << "\n";
    std::cout << "Ending...\n";
    return 0;
}
