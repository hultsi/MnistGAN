#pragma once

#include <fstream>
#include <array>

namespace mnistParser {
    inline constexpr const int IMAGE_PIXELS = 784; // 28x28
    
    inline constexpr const int IMAGE_OFFSET = 16; // 32 * 4 = 128 bits = 16 bytes
    inline constexpr const int LABEL_OFFSET = 8; // 32 * 2 = 64 bits = 8 bytes
    
    inline constexpr const int TRAIN_DATA_SIZE = 47040016;
    inline constexpr const int TRAIN_LABEL_SIZE = 60008;

    inline constexpr const int TEST_DATA_SIZE = 7840016;
    inline constexpr const int TEST_LABEL_SIZE = 10008;

    inline constexpr const int TRAIN_IMAGE_MAX = 60000;
    inline constexpr const int TEST_IMAGE_MAX = 10000;
    
    int flipInt32(int32_t i);
    
    /**
     * TEST SET LABEL FILE (t10k-labels-idx1-ubyte):
     * [offset] [type]          [value]          [description]
     * 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
     * 0004     32 bit integer  10000            number of items
     * 0008     unsigned byte   ??               label
     * 0009     unsigned byte   ??               label
     * ........
     * xxxx     unsigned byte   ??               label
     * The labels values are 0 to 9.
     * 
     * TEST SET IMAGE FILE (t10k-images-idx3-ubyte):
     * [offset] [type]          [value]          [description]
     * 0000     32 bit integer  0x00000803(2051) magic number
     * 0004     32 bit integer  10000            number of images
     * 0008     32 bit integer  28               number of rows
     * 0012     32 bit integer  28               number of columns
     * 0016     unsigned byte   ??               pixel
     * 0017     unsigned byte   ??               pixel
     * ........
     * xxxx     unsigned byte   ??               pixel
     * Pixels are organized row-wise. Pixel values are 0 to 255. 0 means background (white), 255 means foreground (black).
     */
    namespace test {
        extern int32_t currentImageNr; 
        extern std::array<int32_t, IMAGE_PIXELS> currentImageInt;
        extern std::array<float, IMAGE_PIXELS> currentImageFloat;
        
        // Open these streams to point to the proper files
        extern std::ifstream testImgStrm;
        extern std::ifstream testLabelStrm;

        extern std::ofstream outStream;

        std::array<float, IMAGE_PIXELS> getImage(int32_t nr);
        int32_t getImageNr(int32_t nr);
    }

    /**
     * TRAINING SET LABEL FILE (train-labels-idx1-ubyte):
     * [offset] [type]          [value]          [description]
     * 0000     32 bit integer  0x00000801(2049) magic number (MSB first)
     * 0004     32 bit integer  60000            number of items
     * 0008     unsigned byte   ??               label
     * 0009     unsigned byte   ??               label
     * ........
     * xxxx     unsigned byte   ??               label
     * The labels values are 0 to 9.
     * 
     * TRAINING SET IMAGE FILE (train-images-idx3-ubyte):
     * [offset] [type]          [value]          [description]
     * 0000     32 bit integer  0x00000803(2051) magic number
     * 0004     32 bit integer  60000            number of images
     * 0008     32 bit integer  28               number of rows
     * 0012     32 bit integer  28               number of columns
     * 0016     unsigned byte   ??               pixel
     * 0017     unsigned byte   ??               pixel
     * ........
     * xxxx     unsigned byte   ??               pixel
     */
    namespace training {
        extern int32_t currentImageNr; 
        extern std::array<int32_t, IMAGE_PIXELS> currentImageInt;
        extern std::array<float, IMAGE_PIXELS> currentImageFloat;
        
        // Open these streams to point to the proper files
        extern std::ifstream trainImgStrm;
        extern std::ifstream trainLabelStrm;

        extern std::ofstream outStream;

        std::array<float, IMAGE_PIXELS> getImage(int32_t nr);
        int32_t getImageNr(int32_t nr);
    }    
}