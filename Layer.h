#pragma once
#include <vector>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class Layer
{
public:
	void forward();
	void backward();
};


