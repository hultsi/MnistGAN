#pragma once

#include <cmath>

/**
 * Compile time functions
 */
namespace templates {
    /**
     * Calculates natural log to a precision defined by the initial guess yPrev and iterations
     */
    constexpr float logn(float x, float yPrev = 1, int iterations = 100) {
        return (iterations > 0 ?  
                logn(x, yPrev - (std::exp(yPrev) - x) / std::exp(yPrev), iterations - 1) : 
                yPrev - (std::exp(yPrev) - x) / std::exp(yPrev));
    }
}