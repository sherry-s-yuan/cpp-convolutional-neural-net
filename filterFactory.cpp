#include "FilterFactory.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <assert.h>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

Matrix FilterFactory::guassianFilter2D(int height, int width, double sigma)
{
    Matrix kernel(height, Array(width));
    double sum = 0.0;
    int i, j;

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            kernel[i][j] = exp(-(i * i + j * j) / (2 * sigma * sigma)) / (2 * 3.14 * sigma * sigma);
            sum += kernel[i][j];
        }
    }

    for (i = 0; i < height; i++) {
        for (j = 0; j < width; j++) {
            kernel[i][j] /= sum;
        }
    }
    return kernel;
}

Image FilterFactory::guassianFilter3D(int channel, int height, int width, double sigma)
{
    Image kernel(channel, Matrix(height, Array(width)));
    double sum = 0.0;
    int d, i, j;
    for (d = 0; d < channel; d++) {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                kernel[d][i][j] = exp(-(d*d + i * i + j * j) / (2 * sigma * sigma)) / (2 * 3.14 * sigma * sigma);
                sum += kernel[d][i][j];
            }
        }
    }
    for (d = 0; d < channel; d++) {
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                kernel[d][i][j] /= sum;
            }
        }
    }
    return kernel;
}


Matrix FilterFactory::construct2D(string filterType, int height, int width, double sigma) {
    if (filterType.compare("gaussian") == 0) {
        return guassianFilter2D(height, width, sigma);
    }
    else {
        // return guassian filter by default
        return guassianFilter2D(height, width, sigma);
    }
}

Image FilterFactory::construct3D(string filterType, int channel, int height, int width, double sigma) {
    if (filterType.compare("gaussian") == 0) {
        return guassianFilter3D(channel, height, width, sigma);
    }
    else {
        // return guassian filter by default
        return guassianFilter3D(channel, height, width, sigma);
    }

}
