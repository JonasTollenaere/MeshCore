//
// Created by Jonas on 9/09/2025.
//

#ifndef MESHCORE_GIFWRAPPER_H
#define MESHCORE_GIFWRAPPER_H

#include <vector>

#include "src/external/gifencoder/GifEncoder.h"
#include <QImage>

class GifWrapper {

    // Input data
    size_t width;
    size_t height;

    // Inner state
    std::vector<uint8_t> pixels;
    GifEncoder gifEncoder;

public:
    explicit GifWrapper(size_t width, size_t height): width(width), height(height), pixels(width * height * 4, 0){}

    void open(const std::string& fileName) {
        // Initialise the gif encoder
        int quality = 10;
        bool useGlobalColorMap = false;
        int loop = 0;
        int preAllocSize = width * height * 3;
        if (!gifEncoder.open(fileName, width, height, quality, useGlobalColorMap, loop, preAllocSize)) {
            fprintf(stderr, "Error opening the gif file\n");
        }
    }

    void close() {
        if (!gifEncoder.close()) {
            fprintf(stderr, "Error closing gif file\n");
        }
    }

    void appendFrame(const std::vector<uint8_t>& pixels, int delay) {
        gifEncoder.push(GifEncoder::PIXEL_FORMAT_RGBA, pixels.data(), width, height, delay);
    }

    void appendFrame(const QImage& frame, int delay) {
        for (int x = 0; x < width; ++x){
            for (int y = 0; y < height; ++y){
                const auto rgb = frame.pixel(x, y);
                pixels[4 * (y * width + x) + 0] = (rgb >> 16) & 0xFF;
                pixels[4 * (y * width + x) + 1] = (rgb >> 8) & 0xFF;
                pixels[4 * (y * width + x) + 2] = (rgb >> 0) & 0xFF;
                pixels[4 * (y * width + x) + 3] = 255;
            }
        }
        gifEncoder.push(GifEncoder::PIXEL_FORMAT_RGBA, pixels.data(), width, height, delay);
    }
};

#endif //MESHCORE_GIFWRAPPER_H