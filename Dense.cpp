#include "Dense.h"
#include <vector>
#include <png++/png.hpp>
#include <list>
#include "ImageInput.h"
#include "Filter2D.h"
#include "transformations/Dot.h"
using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

Dense::Dense(int inDim, int outDim) {
    weight = Matrix(inDim, Array(outDim));
}
void Dense::saveImage(const char* filename) {

}

void Dense::forward(Matrix input) {
    forwardInput = input;
    Dot dot = Dot();
    forwardOutput = dot.forward(input, weight);
    if (hasOutputLayer)
        outputLayer->forward(forwardOutput);
}
void Dense::backward(Matrix dOutput) {
    backwardInput = dOutput;
    Dot dot = Dot();
    backwardOutput = dot.backwardInput(weight, dOutput);
    Matrix dWeight = dot.backwardInput(forwardInput, dOutput);
    if (hasInputLayer)
        inputLayer->backward(backwardOutput);
}

void Dense::setInputLayer(Layer* in) {
    inputLayer = in;
    hasInputLayer = true;
}
void Dense::setOutputLayer(Layer* out) {
    outputLayer = out;
    hasOutputLayer = true;
}
