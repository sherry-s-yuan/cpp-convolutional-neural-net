#pragma once
#include "Layer.h"
#include <vector>
#include <list>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;


class ConvolutionLayer:public Layer
{
public:
    int channel, height, width, filterSize;
    int skip=1;
    Matrix filter;
    Layer inputLayer;
    Layer outputLayer;
    Image forwardOutput;
    Image backwardOutput;
    // initialize emtpy matrix from given dimension
    ConvolutionLayer(int fs, int s=1); // filter size, stride
    ConvolutionLayer(Image inputImage);

    void saveImage(const char* filename);
    Image forward(Image convIn);
    Image backward(Image convIn);

};

