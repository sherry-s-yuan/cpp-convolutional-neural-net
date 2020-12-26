#pragma once
#include "Transformation.h"
#include <vector>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class Transpose :
    public Transformation
{
public:
    Matrix forward(Matrix input);
    Matrix backward(Matrix dOutput);
};

