#pragma once
#include <vector>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class Layer
{
public:
	virtual void forward(Array arr);
	virtual void backward(Array arr);
	virtual void forward(Matrix mat);
	virtual void backward(Matrix mat);
	virtual void forward(Image img);
	virtual void backward(Image img);
};


