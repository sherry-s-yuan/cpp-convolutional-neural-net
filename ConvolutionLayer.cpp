#include "ConvolutionLayer.h"
#include <vector>
#include <png++/png.hpp>
#include <list>
#include "ImageInput.h"

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;


ConvolutionLayer::ConvolutionLayer(int fs, int s) {
    filterSize = fs;
    skip = s;
    Matrix filter(filterSize, Array(filterSize));
}

ConvolutionLayer::ConvolutionLayer(Image inputImage) {
    forwardOutput = inputImage;
    channel = forwardOutput.size();
    height = forwardOutput[0].size();
    width = forwardOutput[0][0].size();
}

void ConvolutionLayer::saveImage(const char* filename) {
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


Image ConvolutionLayer::forward(Image convIn) {
    int filterHeight = filterSize;
    int filterWidth = filterSize;

    height = (convIn[0].size() - filterHeight) / skip + 1;
    width = (convIn[0][0].size() - filterWidth) / skip + 1;

    Image fOutput(3, Matrix(height, Array(width)));

    int d, i, j, h, w;

    // Image newImage(3, Matrix(newImageHeight, Array(newImageWidth)));
    // for each channel (3) apply the filter to each local receptive field
    for (d = 0; d < 3; d++) {
        // loop through each index of image
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                // apply filter
                int imgRow = i * skip;
                int imgCol = j * skip;
                for (h = imgRow; h < imgRow + filterHeight; h++) {
                    for (w = imgCol; w < imgCol + filterWidth; w++) {
                        fOutput[d][i][j] += filter[h - imgRow][w - imgCol] * convIn[d][h][w];
                    }
                }
            }
        }
    }
    forwardOutput = fOutput;
    return fOutput;

}
Image ConvolutionLayer::backward(Image convIn) {
    Image dConvIn(3, Matrix(convIn[0].size(), Array(convIn[0][0].size())));
    
    int filterHeight = filterSize;
    int filterWidth = filterSize;

    int convOutHeight = backwardOutput[0].size();
    int convOutWidth = backwardOutput[0][0].size();

    // initilize jacobian matrix for filter
    Matrix dFilter = Matrix(filterHeight, Array(filterWidth));
    // initilize jacobian matrix for previous convolution layer (convIn)

    int d, i, j, h, w;

    // for each channel (3) calculate derivative with respect to X and F
    for (d = 0; d < 3; d++) {
        // loop through each index of outputConv's derivative
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                // apply filter
                int imgRow = i * skip;
                int imgCol = j * skip;
                for (h = imgRow; h < imgRow + filterHeight; h++) {
                    for (w = imgCol; w < imgCol + filterWidth; w++) {
                        dFilter[h - imgRow][w - imgCol] += convIn[d][h][w] * backwardOutput[d][i][j];
                        dConvIn[d][h][w] += backwardOutput[d][i][j] * filter[h - imgRow][w - imgCol];
                    }
                }
            }
        }
    }
    // TODO: do something with dFilter
    return dConvIn;
}
