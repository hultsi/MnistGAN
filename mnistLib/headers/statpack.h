#pragma once

#include <array>
#include <vector>
#include <functional>
#include <random>
#include <cmath>
#include <cassert>
#include <iostream>

namespace statpack {
    class Random {
    public:
        static void seed() {
            engine.seed(rd());
        }

        static void seed(const unsigned int value) {
            engine.seed(value);
        }

        static int Int(const int min, const int max) {
            std::uniform_int_distribution<int> dist(min,max);
            return dist(engine);
        }

        static float Float(const float min, const float max) {
            std::uniform_real_distribution<float> dist(min, max);
            return dist(engine);
        }

        static double Double(const double min, const double max) {
            std::uniform_real_distribution<double> dist(min,max);
            return dist(engine);
        }

    private:
        inline static std::random_device rd;
        inline static std::mt19937 engine;
    };
}


/*
 * General templates
 **/
namespace statpack {
    template <typename T>
    struct ImageVector {
        std::vector<T> data;
        int width;
        int height;
    };

    template <typename T, size_t windowSize>
    struct MovingMean {
        std::array<T, windowSize> data{};
        size_t startIndex;
        float mean;

        MovingMean() : startIndex(0), mean(0) {};

        float update(const T newValue) {
            T removed = data[startIndex];
            data[startIndex] = newValue;
            startIndex = ++startIndex % windowSize;

            mean = mean + ((newValue - removed) / windowSize);
            return mean;
        }
    };

    template <typename T>
    T sigmoid(const T x) {
        return static_cast<T>(1) / (static_cast<T>(1) + std::exp(-x));
    }

    template <typename T>
    T dSigmoid(const T x) {
        const T s = sigmoid(x);
        return s * (static_cast<T>(1) - s);
    }

    template <typename T>
    T normalize(const T input, const T min0, const T max0, const T min1, const T max1) {
        return (max1 - min1) / (max0 - min0) * (input - max0) + max1;
    }
}

/**
 *  Vector templates 
 */
namespace statpack {
    template <typename T>
    T mse(const std::vector<T> &observed, const std::vector<T> &predicted) {
        #ifdef CUSTOM_DEBUG
            assert(!(observed.size() != predicted.size()) && "Vector sizes are not equal.");
        #endif
        float mse = 0;
        for (size_t i = 0; i < observed.size(); ++i) {
            mse += std::pow(observed[i] - predicted[i], 2);
        }
        return mse / observed.size();
    }

    template <typename T>
    T dMse(const std::vector<T> &observed, const std::vector<T> &predicted) {
#ifdef CUSTOM_DEBUG
        assert(!(observed.size() != predicted.size()) && "Vector sizes are not equal.");
#endif
        float mse = 0;
        for (size_t i = 0; i < observed.size(); ++i) {
            mse += 2 * (observed[i] - predicted[i]);
        }
        return mse / observed.size();
    }

    template <typename T>
    float weightedSum(const std::vector<T>& inputs, const std::vector<T> &weights) {
#ifdef CUSTOM_DEBUG
        assert(!(inputs.size() != weights.size()) && "Vector sizes are not equal.");
#endif
        float wSum = 0;
        for (size_t i = 0; i < inputs.size(); i++) {
            wSum += weights[i] * inputs[i];
        }
        return wSum;
    }

    template <typename K, int W_OUT, int H_OUT>
    std::array<K, W_OUT*H_OUT> rescaleImage(const std::vector<K> &img, const int width, const int height) {
        std::array<K, W_OUT*H_OUT> tmp{};
        const float xScale = static_cast<float>(W_OUT) / static_cast<float>(width);
        const float yScale = static_cast<float>(H_OUT) / static_cast<float>(height);

        // horizontal rescale
        for (int row = 0; row < height; ++row) {
            for (int pixel = 0; pixel < width; ++pixel) {
                const int thisPixel = std::ceil(xScale * pixel );
                const int nextPixel = std::ceil(xScale * (pixel + 1));
                
                for (int i = thisPixel; i < nextPixel; ++i) {
                    if (i + row *W_OUT >= W_OUT*H_OUT) {
                        break;
                    }
                    tmp.at(i + row * W_OUT) = img.at(pixel + row * width);
                }
            }
        }
        
        std::array<K, W_OUT*H_OUT> out{};
        // vertical rescale
        for (int col = 0; col < W_OUT; ++col) {
            for (int pixel = 0; pixel < height; ++pixel) {
                const int thisPixel = std::ceil(yScale * pixel );
                const int nextPixel = std::ceil(yScale * (pixel + 1));
                
                for (int i = thisPixel; i < nextPixel; ++i) {
                    if (col + i * W_OUT >= W_OUT*H_OUT) {
                        break;
                    }
                    out.at(col + i * W_OUT) = tmp.at(col + pixel * W_OUT);
                }
            }
        }
        return out;
    }
}

/**
 *  Array templates 
 */
namespace statpack {
    template <int T>
    float mean(const std::array<float, T>& inputs) {
        float y = 0;
        for (int i = 0; i < inputs.size(); ++i) {
            y += inputs[i];
        }
        return y / inputs.size();
    }
    
    template <int T>
    float variance(const std::array<float, T>& inputs)  {
        float y = 0;
        float m = mean<T>(inputs);
        for (int i = 0; i < inputs.size(); ++i) {
            y += std::pow(inputs[i] - m, 2);
        }
        return y / inputs.size();
    }

    template <int T>
    std::array<float, T> standardize(const std::array<float, T>& inputs) {
        std::array<float, T> y;
        float mean = mean<T>(inputs);
        float variance = variance<T>(inputs);
        for (int i = 0; i < inputs.size(); i++) {
            y[i] = (inputs[i] - mean) / variance;
        }
        return y;
    }

    template <int T>
    std::array<float, T> normalize(const std::array<float, T> &inputs, const float min0, const float max0, const float min1, const float max1) {
        std::array<float, T> y;
        for (int i = 0; i < inputs.size(); i++) {
            y[i] = (max1 - min1)/(max0 - min0)*(inputs[i] - max0) + max1;
        }
        return y;
    }

    template<int T>
    float weightedSum(const std::array<float, T> &activations, const std::array<float, T> &weight) {
        float weightedSum = 0;
        for (int i = 0; i < activations.size(); i++) {
            weightedSum += weight[i] * activations[i];
        }
        return weightedSum;
    }

    template<typename K, int T>
    int maxValInd(const std::array<K, T> &arr) {
        int ind = 0;
        for (int i = 1; i < arr.size(); ++i) {
            if (arr[ind] < arr[i]) {
                ind = i;
            }
        }
        return ind;
    }

    template<typename K, int T, int M>
    std::array<K, M> rescaleMnistToHalf(const std::array<K, T> &arr) {
        std::array<K, M> out;
        int ind1 = 0;
        int ind2 = 0;
        while (ind2 < M) {
            out[ind2] = (arr[ind1] + arr[ind1+1] + arr[ind1+28] + arr[ind1+29]) / 4;
            if ((ind1 + 2) % 28 == 0) {
                ind1 += 30;
            } else {
                ind1 += 2;
            }  
            ++ind2;
        }
        return out;
    }

    template <typename T, int WIDTH, int HEIGHT>
    ImageVector<T> cropBlackBackground(const std::array<T, WIDTH*HEIGHT>& image) {
        ImageVector<T> imgOut;
        constexpr const int BLACK = 0;
        int xMin = WIDTH;
        int yMin = HEIGHT;
        int xMax = 0;
        int yMax = 0;
        for (int row = 0; row < HEIGHT; ++row) {
            for (int col = 0; col < WIDTH; ++col) {
                if (image.at(col + row*WIDTH) != BLACK) {
                    xMin = (col < xMin ? col : xMin);
                    xMax = (col > xMax ? col : xMax);
                    
                    yMin = (row < yMin ? row : yMin);
                    yMax = (row > yMax ? row : yMax);
                }
            }
        }
        int ind = 0;
        yMax = (yMax < HEIGHT - 1 ? yMax + 1 : yMax);
        xMax = (xMax < WIDTH  - 1 ? xMax + 1 : xMax);
        for (int row = yMin; row < yMax; ++row) {
            for (int col = xMin; col < xMax; ++col) {
                imgOut.data.emplace_back(image.at(col + row*WIDTH));
            }
        }
        imgOut.width = xMax - xMin;
        imgOut.height = yMax - yMin;
        return imgOut;
    }

    template <typename K, int W_IN, int H_IN, int W_OUT, int H_OUT>
    std::array<K, W_OUT*H_OUT> rescaleImage(const std::array<K, W_IN*H_IN>& img) {
        std::array<K, W_OUT*H_OUT> tmp{};
        const float xScale = static_cast<float>(W_OUT) / static_cast<float>(W_IN);
        const float yScale = static_cast<float>(H_OUT) / static_cast<float>(H_IN);

        // horizontal rescale
        for (int row = 0; row < H_IN; ++row) {
            for (int pixel = 0; pixel < W_IN; ++pixel) {
                const int thisPixel = std::ceil(xScale * pixel );
                const int nextPixel = std::ceil(xScale * (pixel + 1));
                
                for (int i = thisPixel; i < nextPixel; ++i) {
                    tmp.at(i + row * W_OUT) = img.at(pixel + row * W_IN);
                }
            }
        }
        
        std::array<K, W_OUT*H_OUT> out{};
        // vertical rescale
        for (int col = 0; col < W_OUT; ++col) {
            for (int pixel = 0; pixel < H_IN; ++pixel) {
                const int thisPixel = std::ceil(yScale * pixel );
                const int nextPixel = std::ceil(yScale * (pixel + 1));
                
                for (int i = thisPixel; i < nextPixel; ++i) {
                    out.at(col + i * W_OUT) = tmp.at(col + pixel * W_OUT);
                }
            }
        }
        return out;
    }
}