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
    nn.learnRate = .3;
    nn.addLayer(3);
    nn.addLayer(4);
    nn.addLayer(8);
    nn.addLayer(1);
    nn.setCostFunction("mse");
    nn.setActivationFunction("sigmoid");
    nn.build();
    nn.randomizeWeightsAndBiases();
    nn.inputMin = 0;
    nn.inputMax = 1;
    nn.targetMin = 0;
    nn.targetMax = 3;
    // todo: this is just test setup
    std::vector<std::vector<std::vector<float>>> inputs {
        { {0,0,0}, {0} },
        { {0,1,1}, {2} },
        { {0,0,1}, {1} },
        { {1,1,1}, {3} },
        { {0,0,1}, {1} },
        { {1,1,0}, {2} },
        { {0,1,1}, {2} },
        { {0,0,1}, {1} },
        { {0,1,1}, {2} },
        { {1,0,1}, {2} }
    };
    constexpr size_t epoch = 10;
    while (true) {
        // Run N number of epochs and check loss
        constexpr size_t N = 30;
        float loss = 0;
        for (size_t k = 0; k < N; ++k) {
            // Run over one epoch set
            for (size_t i = 0; i < epoch; ++i) {
                const int ind = statpack::Random::Int(0, 9);
                loss += nn.train(inputs[ind][0], inputs[ind][1], static_cast<float>(epoch));
            }
            nn.applyDeltas();
        }
        std::cout << "Loss: " << loss << "\n";
        // todo: Do stuff
        if (loss / (N * epoch) < .00001) {
            break;
        }
    }
    std::cout << /*std::round*/(nn.test({0,0,0})[0]*3) << "\n";
    std::cout << /*std::round*/(nn.test({0,1,1})[0]*3) << "\n";
    std::cout << /*std::round*/(nn.test({1,1,1})[0]*3) << "\n";
    std::cout << /*std::round*/(nn.test({1,0,1})[0]*3) << "\n";

    std::cout << "Ending...\n";
    return 0;
}
