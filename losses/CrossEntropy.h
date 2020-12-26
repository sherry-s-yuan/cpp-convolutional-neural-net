#pragma once
#include "Loss.h"
class CrossEntropy :
    public Loss
{
public:
	Matrix yPred;
	Matrix yTrue;
	double forwardOutput;
	Matrix backwardOutput;
	void forward(Matrix yPred, Matrix yTrue);
	void backward();
};

