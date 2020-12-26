#pragma once
#include <vector>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;
class Loss
{
public:
	Matrix yPred;
	Matrix yTrue;
	double forwardOutput;
	Matrix backwardOutput;
	virtual void forward(Matrix yPred, Matrix yTrue);
	virtual void backward();
};

