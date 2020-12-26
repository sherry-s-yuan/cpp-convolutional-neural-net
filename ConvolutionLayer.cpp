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


ConvolutionLayer::ConvolutionLayer(int fs, int ic, int oc, int s) {
    inChannel = ic;
    outChannel = oc;
    skip = s;
    filter = Filter3D(ic, oc, fs, "gaussian");
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
    forwardInput = convIn;

    forwardOutput = filter.forward(convIn, skip);
    channel = forwardOutput.size();
    height = forwardOutput[0].size();
    width = forwardOutput[0][0].size();
    if(hasOutputLayer)
        outputLayer->forward(forwardOutput);
}

void ConvolutionLayer::backward(Image dConvOut) {
    backwardInput = dConvOut;
    
    int filterHeight = filter.filterSize;
    int filterWidth = filter.filterSize;
    int filterChannel = filter.filterChannel;

    int convOutHeight = dConvOut[0].size();
    int convOutWidth = dConvOut[0][0].size();
    int convOutChannel = dConvOut.size();

    // initilize jacobian matrix for previous convolution layer (convIn)
    Image dConvIn(forwardInput.size(), Matrix(forwardInput[0].size(), Array(forwardInput[0][0].size())));

    int d, i, j, h, w, c;
    filter.backward(dConvOut, forwardInput, skip);

    // for each channel (3) calculate derivative with respect to X and F
    for (d = 0; d < convOutChannel; d++) {
        // loop through each index of image
        for (i = 0; i < convOutHeight; i++) {
            for (j = 0; j < convOutWidth; j++) {
                // apply filter
                int imgCnl = d;
                int imgRow = i * skip;
                int imgCol = j * skip;
                for (c = imgCnl; c < imgCnl + filterChannel; c++) {
                    for (h = imgRow; h < imgRow + filterHeight; h++) {
                        for (w = imgCol; w < imgCol + filterWidth; w++) {
                            dConvIn[c][h][w] += filter.filter[c - imgCnl][h - imgRow][w - imgCol] * dConvOut[d][i][j];
                        }
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
