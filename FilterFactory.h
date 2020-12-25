#pragma once
#include <vector>
#include <cmath>
#include <iostream>
#include <assert.h>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class FilterFactory
{
public:
	Matrix guassianFilter2D(int size, double sigma);
	Matrix construct2D(string filterType, int size, double sigma = 1);
};

