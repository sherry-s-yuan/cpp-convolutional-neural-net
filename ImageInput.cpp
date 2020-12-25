#include "ImageInput.h"
#include <vector>
#include <png++/png.hpp>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;


ImageInput::ImageInput(const char* filename) {
    forwardOutput = loadImage(filename);
    channel = forwardOutput.size();
    height = forwardOutput[0].size();
    width = forwardOutput[0][0].size();
    
}

ImageInput::ImageInput(Image inputImage) {
    forwardOutput = inputImage;
    channel = forwardOutput.size();
    height = forwardOutput[0].size();
    width = forwardOutput[0][0].size();
}

Image ImageInput::loadImage(const char* filename) {
    png::image<png::rgb_pixel> image(filename);
    Image imageMatrix(3, Matrix(image.get_height(), Array(image.get_width())));

    int h, w;
    for (h = 0; h < image.get_height(); h++) {
        for (w = 0; w < image.get_width(); w++) {
            imageMatrix[0][h][w] = image[h][w].red;
            imageMatrix[1][h][w] = image[h][w].green;
            imageMatrix[2][h][w] = image[h][w].blue;
        }
    }
    return imageMatrix;
}

void ImageInput::saveImage(const char* filename) {
    int x, y;

    png::image<png::rgb_pixel> imageFile(width, height);

    for (y = 0; y < height; y++) {
        for (x = 0; x < width; x++) {
            imageFile[y][x].red = forwardOutput[0][y][x];
            imageFile[y][x].green = forwardOutput[1][y][x];
            imageFile[y][x].blue = forwardOutput[2][y][x];
        }
    }
    imageFile.write(filename);
}

