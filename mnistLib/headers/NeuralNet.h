#pragma once

#include <fstream>
#include <array>
#include <vector>
#include <string>
#include <tuple>

#include "statpack.h"
#include "mnistParser.h"
#include "variadics.h"

namespace mnistGAN {
    template <int FIRST, int... REST>
    class NeuralNet {
    public:
        static constexpr int LAYERS = 1 + sizeof...(REST);
        static constexpr int WEIGHTS = LAYERS - 1;
        static constexpr int BIASES = LAYERS - 1;

        std::array<float, variadics::Sum<FIRST, REST...>::value> nodes;
        static constexpr int test = variadics::Last<FIRST, REST...>::value;
        NeuralNet() {}
    };
}

// template <int... SIZE>
// class NeuralNet {
// public:
//     using array_tuple = std::tuple<std::array<float,SIZE>...>;

//     static constexpr int LAYERS = std::tuple_size<array_tuple>{};
//     static constexpr int WEIGHTS = LAYERS - 1;
//     static constexpr int BIASES = LAYERS - 1;

//     array_tuple layers{};
//     typename tupleRemove::last<array_tuple>::type weights;
//     typename tupleRemove::last<array_tuple>::type biases;

//     NeuralNet() {}
// };