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
	Matrix guassianFilter2D(int height, int width, double sigma);
	Image guassianFilter3D(int channel, int height, int width, double sigma);
	Matrix construct2D(string filterType, int height, int width, double sigma = 1);
	Image construct3D(string filterType, int channel, int height, int width, double sigma = 1);
};

