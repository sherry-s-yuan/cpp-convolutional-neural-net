#include "Transformation.h"
#include <vector>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

//Image imageMatrix(3, Matrix(image.get_height(), Array(image.get_width())));

Array Transformation::forward(Array arr) { return Array(1); }
Array Transformation::backward(Array arr) { return Array(1); }
Matrix Transformation::forward(Matrix mat) { return Matrix(1, Array(1)); }
Matrix Transformation::backward(Matrix mat) { return Matrix(1, Array(1)); }
Image Transformation::forward(Image img) { return Image(1, Matrix(1, Array(1))); }
Image Transformation::backward(Image img) { return Image(1, Matrix(1, Array(1))); }


Matrix Transformation::forward(Matrix mat, Matrix mat2) { return Matrix(1, Array(1)); }
Matrix Transformation::backward(Matrix mat, Matrix mat2) { return Matrix(1, Array(1)); }