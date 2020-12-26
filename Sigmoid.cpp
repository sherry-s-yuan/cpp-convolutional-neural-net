#include "Sigmoid.h"
#include <vector>
using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;




void Sigmoid::forward(Matrix input) {
	forwardInput = input;
	Matrix transformed = Matrix(forwardInput.size(), Array(forwardInput[0].size()));
	int i, j;
	for (i = 0; i < forwardInput.size(); i++) {
		for (j = 0; j < forwardInput[0].size(); j++) {
			transformed[i][j] = 1.0 / (1.0 + exp(-forwardInput[i][j]));
		}
	}
	forwardOutput = transformed;
	if (hasOutputLayer)
		outputLayer->forward(forwardOutput);
}

void Sigmoid::backward(Matrix dOutput) {
	backwardInput = dOutput;
	Matrix derivative = Matrix(forwardOutput.size(), Array(forwardOutput[0].size()));
	int i, j;
	for (i = 0; i < forwardOutput.size(); i++) {
		for (j = 0; j < forwardOutput[0].size(); j++) {
			derivative[i][j] = forwardOutput[i][j] * (1 - forwardOutput[i][j]);
		}
	}
	backwardOutput = derivative;
	if (hasInputLayer)
		inputLayer->backward(backwardOutput);
}

void Sigmoid::setInputLayer(Layer* in) {
    inputLayer = in;
    hasInputLayer = true;
}
void Sigmoid::setOutputLayer(Layer* out) {
    outputLayer = out;
    hasOutputLayer = true;
}
