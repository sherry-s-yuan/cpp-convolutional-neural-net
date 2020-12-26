#pragma once
#include "Layer.h"
class Sigmoid :
    public Layer
{
public:
    Layer* inputLayer;
    Layer* outputLayer;
    Matrix forwardInput;
    Matrix forwardOutput;
    Matrix backwardOutput;
    Matrix backwardInput;
    bool hasInputLayer = false;
    bool hasOutputLayer = false;

    // initialize emtpy matrix from given dimension
    void forward(Matrix convIn);
    void backward(Matrix dConvOut);

    void setInputLayer(Layer* in);
    void setOutputLayer(Layer* out);
};

