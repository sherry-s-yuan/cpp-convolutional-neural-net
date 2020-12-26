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
    Dot dot = Dot();
    forwardOutput = dot.forward(input, weight);
}
void Dense::backward(Matrix dOutput) {

}

void Dense::setInputLayer(Layer* in) {
    inputLayer = in;
    hasInputLayer = true;
}
void Dense::setOutputLayer(Layer* out) {
    outputLayer = out;
    hasOutputLayer = true;
}
