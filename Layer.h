#pragma once
#include <vector>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class Layer
{
public:
    virtual void forward(Image convIn);
    virtual void backward(Image convIn);
};


