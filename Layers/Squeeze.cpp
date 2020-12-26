#include "Squeeze.h"
#include <vector>
#include <png++/png.hpp>
#include <list>
#include "../ImageInput.h"
#include "../Links/Filter2D.h"
using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

Squeeze::Squeeze(int dim) {
    squeezeDimension = dim;
}

void Squeeze::forward(Image convIn) {
    forwardInput = convIn;
    if (squeezeDimension == 0) {
        forwardOutput = convIn[0];        
    }
    else if (squeezeDimension == 1) {
        Matrix out = Matrix(convIn.size(), Array(convIn[0][0].size()));
        int i, k;
        for (i = 0; i < convIn.size(); i++)
            for (k = 0; k < convIn[0][0].size(); k++)
                out[i][k] = convIn[i][0][k];
        forwardOutput = out;
    }
    else if (squeezeDimension == 2) {
        Matrix out = Matrix(convIn.size(), Array(convIn[0].size()));
        int i, j;
        for (i = 0; i < convIn.size(); i++)
            for (j = 0; j < convIn[0].size(); j++)
                out[i][j] = convIn[i][j][0];
        forwardOutput = out;
    }
    else {
        forwardOutput = convIn[0];
    }

    if (hasOutputLayer)
        outputLayer->forward(forwardOutput);

}

void Squeeze::backward(Matrix dConvOut) {
    backwardInput = dConvOut;
    if (squeezeDimension == 0) {
        backwardOutput.push_back(dConvOut);
    }
    else if (squeezeDimension == 1) {
        int i;
        for (i = 0; i < dConvOut.size(); i++) {
            Matrix tmp;
            tmp.push_back(dConvOut[i]);
            backwardOutput.push_back(tmp);
        }
    }
    else if (squeezeDimension == 2) {
        int i, j;
        for (i = 0; i < dConvOut.size(); i++)
            for (j = 0; j < dConvOut[0].size(); j++)
                backwardOutput[i][j][0] = dConvOut[i][j];
    }
    else {
        backwardOutput.push_back(dConvOut);
    }

    if (hasInputLayer)
        inputLayer->backward(backwardOutput);
}

void Squeeze::setInputLayer(Layer* in) {
    inputLayer = in;
    hasInputLayer = true;
}
void Squeeze::setOutputLayer(Layer* out) {
    outputLayer = out;
    hasOutputLayer = true;
}
