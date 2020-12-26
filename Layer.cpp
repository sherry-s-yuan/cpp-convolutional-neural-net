#include "Layer.h"
#include <vector>

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

void Layer::forward(Array arr) {}
void Layer::backward(Array arr) {}
void Layer::forward(Matrix mat) {}
void Layer::backward(Matrix mat) {}
void Layer::forward(Image img) {}
void Layer::backward(Image img) {}



