#include "ConvolutionLayer.h"
#include <vector>
#include <png++/png.hpp>
#include <list>
#include "ImageInput.h"
#include "Filter2D.h"
using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;


ConvolutionLayer::ConvolutionLayer(int fs, int s) {
    skip = s;
    filter = Filter2D(fs, "Gaussian");
    //Matrix filter(filterSize, Array(filterSize));
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


void ConvolutionLayer::forward(Image convIn) {
    int filterHeight = filter.filterSize; 
    int filterWidth = filter.filterSize;

    forwardInput = convIn;

    forwardOutput = filter.forward(convIn, skip);
    channel = forwardOutput.size();
    height = forwardOutput[0].size();
    width = forwardOutput[0][0].size();
    if(hasOutputLayer)
        outputLayer->forward(forwardOutput);
}

void ConvolutionLayer::backward(Image dConvOut) {
    Image dConvIn(3, Matrix(forwardInput[0].size(), Array(forwardInput[0][0].size())));
    backwardInput = dConvOut;
    
    int filterHeight = filter.filterSize;
    int filterWidth = filter.filterSize;

    int convOutHeight = dConvOut[0].size();
    int convOutWidth = dConvOut[0][0].size();

    // initilize jacobian matrix for filter
    Matrix dFilter = Matrix(filterHeight, Array(filterWidth));
    // initilize jacobian matrix for previous convolution layer (convIn)

    int d, i, j, h, w;
    filter.backward(dConvOut, forwardInput, skip);

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
                        // dFilter[h - imgRow][w - imgCol] += convIn[d][h][w] * backwardOutput[d][i][j];
                        dConvIn[d][h][w] += dConvOut[d][i][j] * filter.filter[h - imgRow][w - imgCol];
                    }
                }
            }
        }
    }
    backwardOutput = dConvIn;
    if (hasInputLayer)
        inputLayer->backward(backwardOutput);
}

void ConvolutionLayer::setInputLayer(Layer* in) {
    inputLayer = in;
    hasInputLayer = true;
}
void ConvolutionLayer::setOutputLayer(Layer* out) {
    outputLayer = out;
    hasOutputLayer = true;
}
