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
    NeuralNet nn;
    nn.addLayer(4);
    nn.addLayer(2);
    nn.build();
    std::cout << "Ending...\n";
    return 0;
}
