#include <iostream>
#include <vector>
#include <assert.h>
#include <cmath>
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

    // Image image = loadImage("image.png");
    cout << "Applying filter..." << endl;
    convLayer1.forward(image.forwardOutput);

    convLayer3.backward(convLayer3.forwardOutput);
    // convLayer.forwardOutput = convOut;
    cout << "Saving image..." << endl;
    convLayer3.saveImage("newImage.png");
    cout << "Done!" << endl;
}

