#pragma once
#include <vector>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class Transformation
{
public:
	virtual Array forward(Array arr);
	virtual Array backward(Array arr);
	virtual Matrix forward(Matrix mat);
	virtual Matrix backward(Matrix mat);
	virtual Matrix forward(Matrix mat, Matrix mat2);
	virtual Matrix backward(Matrix mat, Matrix mat2);
	virtual Image forward(Image img);
	virtual Image backward(Image img);
};

