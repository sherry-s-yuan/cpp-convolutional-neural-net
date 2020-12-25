#pragma once
#include "Filter.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <assert.h>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class Filter2D :
    public Filter
{
public:
    Matrix filter;
    Matrix dFilter;
    int filterSize;
    Filter2D(int fs, string filterType);
    Filter2D();
    Image forward(Image convIn, int skip);
    Matrix backward(Image dConvOut, Image convIn, int skip);
};

