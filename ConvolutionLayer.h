#pragma once
#include "Layer.h"
#include <vector>
#include <list>
#include "Filter2D.h"

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;


class ConvolutionLayer:public Layer
{
public:
    int channel, height, width, filterSize;
    int skip=1;
    Filter2D filter;
    Layer inputLayer;
    Layer outputLayer;
    Image forwardOutput;
    Image backwardOutput;
    // initialize emtpy matrix from given dimension
    ConvolutionLayer(int fs, int s=1); // filter size, stride

    void saveImage(const char* filename);
    Image forward(Image convIn);
    Image backward(Image convIn);

};

