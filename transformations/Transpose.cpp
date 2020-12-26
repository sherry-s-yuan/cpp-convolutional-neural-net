#include "Transpose.h"
#include <vector>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

Matrix Transpose::forward(Matrix input) {
	Matrix output = Matrix(input[0].size(), Array(input.size()));
	int i, j;
	for (i = 0; i < input.size(); i++)
		for (j = 0; j < input[0].size(); j++)
			output[j][i] = input[i][j];
	return output;
}

Matrix Transpose::backward(Matrix dOutput) {
	Matrix dInput = Matrix(dOutput[0].size(), Array(dOutput.size()));
	int i, j;
	for (i = 0; i < dOutput.size(); i++)
		for (j = 0; j < dOutput[0].size(); j++)
			dInput[j][i] = dOutput[i][j];
	return dInput;
}