#include "Filter2D.h"
#include "FilterFactory.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <assert.h>


Filter2D::Filter2D() {
    FilterFactory filterFactory = FilterFactory();
    filter = filterFactory.construct2D("gaussian", 5);
    filterSize = 5;
}

Filter2D::Filter2D(int fs, string filterType) {
	FilterFactory filterFactory = FilterFactory();
	filter = filterFactory.construct2D(filterType, fs);
	filterSize = fs;
}

Image Filter2D::forward(Image convIn, int skip) {
    int filterHeight = filterSize;
    int filterWidth = filterSize;

    int height = (convIn[0].size() - filterHeight) / skip + 1;
    int width = (convIn[0][0].size() - filterWidth) / skip + 1;

    Image fOutput(3, Matrix(height, Array(width)));

    int d, i, j, h, w;

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
    return fOutput;
}

Matrix Filter2D::backward(Image dConvOut, Image convIn, int skip) {
    int filterHeight = filter.size();
    int filterWidth = filter[0].size();

    int convOutHeight = dConvOut[0].size();
    int convOutWidth = dConvOut[0][0].size();

    // initilize jacobian matrix for filter
    Matrix df = Matrix(filterHeight, Array(filterWidth));
    // initilize jacobian matrix for previous convolution layer (convIn)

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
                        df[h - imgRow][w - imgCol] += convIn[d][h][w] * dConvOut[d][i][j];
                    }
                }
            }
        }
    }
    dFilter = df;
    return df;
}