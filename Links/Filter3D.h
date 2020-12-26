#pragma once
#include "Filter.h"
#include "FilterFactory.h"
#include <vector>
#include <cmath>
#include <iostream>
#include <assert.h>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class Filter3D :
    public Filter
{
public:
    Image filter;
    Image dFilter;
    int filterSize;
    int filterChannel;
    int inChannel, outChannel;
    Filter3D(int ic, int oc, int fs, string filterType);
    Filter3D();
    Image forward(Image convIn, int skip);
    Image backward(Image dConvOut, Image convIn, int skip);

};

