#include "Dot.h"
#include <vector>
#include "Transpose.h"

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;


Matrix Dot::forward(Matrix input, Matrix weight) {
	Matrix output = Matrix(input.size(), Array(weight[0].size()));
    int i, j, k;
    for (i = 0; i < input.size(); ++i)
        for (j = 0; j < weight[0].size(); ++j)
            for (k = 0; k < input[0].size(); ++k)
            {
                output[i][j] += input[i][k] * weight[k][j];
            }
    return output;
}

Matrix Dot::backwardWeight(Matrix input, Matrix dOutput) {
    return forward(input, dOutput);
}

Matrix Dot::backwardInput(Matrix weight, Matrix dOutput) {
    Transpose transpose = Transpose();
    return forward(dOutput, transpose.forward(weight));
}
