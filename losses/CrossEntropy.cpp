#include "CrossEntropy.h"

void CrossEntropy::forward(Matrix yP, Matrix yT) {
	yPred = yP;
	yTrue = yT;
	int i, j;
	int count = 0;
	double sum = 0;
	for (i = 0; i < yPred.size(); i++) {
		for (j = 0; j < yPred[0].size(); j++) {
			sum += (1 - yTrue[i][j]) * log(1 - yPred[i][j]) + (yTrue[i][j]) * log(yPred[i][j]);
			count++;
		}
	}
	forwardOutput = -sum / (double)count;
}

void CrossEntropy::backward() {
	Matrix derivative = Matrix(yPred.size(), Array(yPred[0].size()));
	int i, j;
	for (i = 0; i < yPred.size(); i++) {
		for (j = 0; j < yPred[0].size(); j++) {
			derivative[i][j] = (yPred[i][j] - yTrue[i][j]) / (yPred[i][j] * (1 - yPred[i][j]));
		}
	}
	backwardOutput = derivative;
}