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

class Dense :
    public Layer
{
public:
    int inputDim, outputDim;
    Matrix weight;
    Layer* inputLayer;
    Layer* outputLayer;
    Matrix forwardInput;
    Matrix forwardOutput;
    Matrix backwardOutput;
    Matrix backwardInput;
    bool hasInputLayer = false;
    bool hasOutputLayer = false;
    // initialize emtpy matrix from given dimension
    Dense(int inDim, int outDim); // filter size, stride
    void saveImage(const char* filename);

    void forward(Matrix input);
    void backward(Matrix dOutput);

    void setInputLayer(Layer* in);
    void setOutputLayer(Layer* out);
};

