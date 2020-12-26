#pragma once
#include "Layer.h"
#include <vector>
#include <png++/png.hpp>
#include <list>
#include "../ImageInput.h"
#include "../Links/Filter2D.h"
using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class Squeeze :
    public Layer
{
public:
    int squeezeDimension;
    Layer* inputLayer;
    Layer* outputLayer;
    Image forwardInput;
    Matrix forwardOutput;
    Image backwardOutput;
    Matrix backwardInput;
    bool hasInputLayer = false;
    bool hasOutputLayer = false;
    // initialize emtpy matrix from given dimension
    Squeeze(int dim); // filter size, stride

    void forward(Image convIn);
    void backward(Matrix dConvOut);

    void setInputLayer(Layer* in);
    void setOutputLayer(Layer* out);
};

