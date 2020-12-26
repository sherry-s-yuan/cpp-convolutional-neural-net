#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
#include "Dense.h"
#include "ImageInput.h"
#include "ConvolutionLayer.h"

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;


int main() {
    std::cout << "Load Image" << endl;
    ImageInput image = ImageInput("image.png");
    std::cout << "Construct Filter" << endl;
    ConvolutionLayer convLayer1 = ConvolutionLayer(10, 2);
    ConvolutionLayer convLayer2 = ConvolutionLayer(5, 1);
    ConvolutionLayer convLayer3 = ConvolutionLayer(5, 1);

    convLayer1.setOutputLayer(&convLayer2);
    convLayer2.setOutputLayer(&convLayer3);
    convLayer3.setInputLayer(&convLayer2);
    convLayer2.setInputLayer(&convLayer1);

    Dense dense1 = Dense(400, 5);
    Dense dense2 = Dense(5, 2);
    Dense dense3 = Dense(2, 1);

    dense1.setOutputLayer(&dense2);
    dense2.setOutputLayer(&dense3);
    dense3.setInputLayer(&dense2);
    dense2.setInputLayer(&dense1);

    // Image image = loadImage("image.png");
    cout << "Dense Forward and Back..." << endl;
    Matrix derivative = Matrix(400, Array(1));
    dense1.forward(image.forwardOutput[0]);
    dense3.backward(derivative);

    cout << "Applying filter..." << endl;
    convLayer1.forward(image.forwardOutput);
    convLayer3.backward(convLayer3.forwardOutput);
    // convLayer.forwardOutput = convOut;
    cout << "Saving image..." << endl;
    convLayer3.saveImage("newImage.png");
    cout << "Done!" << endl;
}

