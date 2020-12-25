#include "FilterFactory.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <assert.h>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

Matrix FilterFactory::guassianFilter2D(int size, double sigma)
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

Matrix FilterFactory::construct2D(string filterType, int size, double sigma) {
    if (filterType.compare("gaussian") == 0) {
        return guassianFilter2D(size, sigma);
        
    }
    else {
        // return guassian filter by default
        return guassianFilter2D(size, sigma);
    }
}
