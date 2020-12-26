#pragma once
#include "Transformation.h"
#include <vector>
#include "Transpose.h"

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;
class Dot :
    public Transformation
{
public:
    Matrix forward(Matrix input, Matrix weight);
    Matrix backwardWeight(Matrix input, Matrix dOutput);
    Matrix backwardInput(Matrix weight, Matrix dOutput);
};

