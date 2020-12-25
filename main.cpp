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
    ConvolutionLayer convLayer = ConvolutionLayer(10, 2);
    // Image image = loadImage("image.png");
    cout << "Applying filter..." << endl;
    Image convOut = convLayer.forward(image.forwardOutput);
    // convLayer.forwardOutput = convOut;
    cout << "Saving image..." << endl;
    convLayer.saveImage("newImage.png");
    cout << "Done!" << endl;
}

