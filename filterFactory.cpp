#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include "ImageInput.h"
#include "ConvolutionLayer.h"

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;


Matrix guassianFilter(int size, double sigma)
{
    Matrix kernel(size, Array(size));
    double sum = 0.0;
    int i, j;

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            kernel[i][j] = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * 3.14 * sigma * sigma);
            sum += kernel[i][j];
        }
    }

    for (i = 0; i < size; i++) {
        for (j = 0; j < size; j++) {
            kernel[i][j] /= sum;
        }
    }

    return kernel;
}

Matrix filterFactory(string filterType, int size, double sigma = 1) {
    if (filterType.compare("gaussian") == 0) {
        return guassianFilter(size, sigma);
    }
    else {
        // return guassian filter by default
        return guassianFilter(size, sigma);
    }
}




Image convolutionForward(Image& image, Matrix& filter, int skip=2) {
    assert(image.size() == 3 && filter.size() != 0);

    int height = image[0].size();
    int width = image[0][0].size();
    int filterHeight = filter.size();
    int filterWidth = filter[0].size();
    int newImageHeight = (height - filterHeight) / skip + 1; // height - filterHeight + 1;
    int newImageWidth = (width - filterWidth) / skip + 1; // width - filterWidth + 1;
    int d, i, j, h, w;

    Image newImage(3, Matrix(newImageHeight, Array(newImageWidth)));
    // for each channel (3) apply the filter to each local receptive field
    for (d = 0; d < 3; d++) {
        // loop through each index of image
        for (i = 0; i < newImageHeight; i++) {
            for (j = 0; j < newImageWidth; j++) {
                // apply filter
                int imgRow = i * skip;
                int imgCol = j * skip;
                for (h = imgRow; h < imgRow + filterHeight; h++) {
                    for (w = imgCol; w < imgCol + filterWidth; w++) {
                        newImage[d][i][j] += filter[h - imgRow][w - imgCol] * image[d][h][w];
                    }
                }
            }
        }
    }

    return newImage;
}

Image convolutionBackward(Image& dConvOut, Image& convIn, Matrix& filter, int skip = 2) {
    assert(dConvOut.size() == 3 && filter.size() != 0);
    
    int filterHeight = filter.size();
    int filterWidth = filter[0].size();

    int convOutHeight= dConvOut[0].size();
    int convOutWidth = dConvOut[0][0].size();
    
    // initilize jacobian matrix for filter
    Matrix dFilter = Matrix(filterHeight, Array(filterWidth));
    // initilize jacobian matrix for previous convolution layer (convIn)
    Image dConvIn(3, Matrix(convIn[0].size(), Array(convIn[0][0].size())));
    
    int d, i, j, h, w;

    // for each channel (3) calculate derivative with respect to X and F
    for (d = 0; d < 3; d++) {
        // loop through each index of outputConv's derivative
        for (i = 0; i < dConvOut[0].size(); i++) {
            for (j = 0; j < dConvOut[0][0].size(); j++) {
                // apply filter
                int imgRow = i * skip;
                int imgCol = j * skip;
                for (h = imgRow; h < imgRow + filterHeight; h++) {
                    for (w = imgCol; w < imgCol + filterWidth; w++) {
                        dFilter[h - imgRow][w - imgCol] += convIn[d][h][w] * dConvOut[d][i][j];
                        dConvIn[d][h][w] += dConvOut[d][i][j] * filter[h - imgRow][w - imgCol];
                    }
                }
            }
        }
    }
    return dConvIn;
}

int main() {
	std::cout << "Construct Filter" << endl;
    Matrix filter = filterFactory("gaussian", 10, 0.5);
	std::cout << "Load Image" << endl;
    ImageInput image = ImageInput("image.png");
    ConvolutionLayer convLayer = ConvolutionLayer(10, 2);
    convLayer.filter = filter;
    // Image image = loadImage("image.png");
    cout << "Applying filter..." << endl;
    Image convOut = convLayer.forward(image.forwardOutput);
    // convLayer.forwardOutput = convOut;
    cout << "Saving image..." << endl;
    convLayer.saveImage("newImage.png");
    cout << "Done!" << endl;
}

