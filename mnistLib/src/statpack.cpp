#include <cassert>
#include <iostream>
#include "statpack.h"

namespace statpack {
    float weightedSum(std::vector<float> inputs, std::vector<float> weights) {
#ifdef CUSTOM_DEBUG
        assert(!(inputs.size() != weights.size()) && "Vector sizes are not equal.");
#endif
        float wSum = 0;
        for (size_t i = 0; i < inputs.size(); i++) {
            wSum += weights[i] * inputs[i];
        }
        return wSum;
    }
}