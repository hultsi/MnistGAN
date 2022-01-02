#pragma once

#include <fstream>
#include <array>
#include <vector>
#include <string>
#include <tuple>

#include "statpack.h"
#include "mnistParser.h"

namespace mnistGAN {
    template <int... SIZE>
    class NeuralNet {
    public:
        using local_tuple = std::tuple<std::array<float,SIZE>...>;

        static constexpr int LAYERS = std::tuple_size<local_tuple>{};
        local_tuple layer;
        
        NeuralNet() {}
        
        // void resizeLayer(int layer, int size) {
        //     layers[layer].arr.resize(size);
        // }
    };
}