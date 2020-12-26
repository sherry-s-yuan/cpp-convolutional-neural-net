#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include "Layers/Dense.h"
#include "Layers/ConvolutionLayer.h"
#include "Layers/Squeeze.h"
#include "Layers/Sigmoid.h"
#include "Losses/CrossEntropy.h"

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;


int main() {
    std::cout << "Load Image" << endl;
    ImageInput image = ImageInput("image.png");
    std::cout << "Construct Filter" << endl;
    ConvolutionLayer convLayer1 = ConvolutionLayer(10, 3, 2, 1);
    ConvolutionLayer convLayer2 = ConvolutionLayer(5, 2, 1, 1);
    ConvolutionLayer convLayer3 = ConvolutionLayer(5, 1, 1, 1);
    Squeeze squeeze = Squeeze(0);
    Dense dense1 = Dense(383, 100);
    Dense dense2 = Dense(100, 10);
    Dense dense3 = Dense(10, 1);
    Sigmoid sigmoid = Sigmoid();
    CrossEntropy ce = CrossEntropy();


    convLayer1.setOutputLayer(&convLayer2);
    convLayer2.setInputLayer(&convLayer1);

    convLayer2.setOutputLayer(&convLayer3);
    convLayer3.setInputLayer(&convLayer2);

    convLayer3.setOutputLayer(&squeeze);
    squeeze.setInputLayer(&convLayer3);

    squeeze.setOutputLayer(&dense1);
    dense1.setInputLayer(&squeeze);

    dense1.setOutputLayer(&dense2);
    dense2.setInputLayer(&dense1);

    dense2.setOutputLayer(&dense3);
    dense3.setInputLayer(&dense2);

    dense3.setOutputLayer(&sigmoid);
    sigmoid.setInputLayer(&dense3);

    

    cout << "Forward Computation..." << endl;
    convLayer1.forward(image.forwardOutput);
    cout << "Backward Computation..." << endl;
    Matrix yTrue = Matrix(383, Array(1));
    yTrue[0][0] = 1;
    ce.forward(sigmoid.forwardOutput, yTrue);
    ce.backward();
    sigmoid.backward(ce.backwardOutput);
    // Matrix derivative = Matrix(383, Array(1));
    // dense3.backward(derivative);
    cout << "Done!" << endl;
}

