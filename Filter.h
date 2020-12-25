#pragma once
#include <vector>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class Filter
{
public:
	Matrix filter;
	void forward(Image convIn);
	void backward(Image convIn);
};

