#pragma once
#include "Layer.h"
#include <vector>
#include <list>
#include "../Links/Filter2D.h"
#include "../Links/Filter3D.h"

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;


class ConvolutionLayer:public Layer
{
public:
    int channel, height, width, filterSize;
    int inChannel, outChannel;
    int skip=1;
    Filter3D filter;
    Layer* inputLayer;
    Layer* outputLayer;
    Image forwardInput;
    Image forwardOutput;
    Image backwardOutput;
    Image backwardInput;
    bool hasInputLayer = false;
    bool hasOutputLayer = false;
    // initialize emtpy matrix from given dimension
    ConvolutionLayer(int fs, int ic, int oc, int s=1); // filter size, stride
    void saveImage(const char* filename);
    void forward(Image convIn);
    void backward(Image dConvOut);

    void setInputLayer(Layer* in);
    void setOutputLayer(Layer* out);

};

