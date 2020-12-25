#pragma once
#include "Layer.h"
#include <vector>
#include "Filter2D.h"

using namespace std;

typedef vector<double> Array;
typedef vector<Array> Matrix;
typedef vector<Matrix> Image;

class ImageInput:public Layer
{
public:
    int channel, height, width;
    int skip = 1;
    Filter2D filter;
    Layer inputLayer;
    Layer outputLayer;
    Image forwardOutput;
    Image backwardOutput;
    // construct from filename

    ImageInput(const char* filename);
    // construct from image matrix
    ImageInput(Image inputImage);
    Image loadImage(const char* filename);
    void saveImage(const char* filename);

    void forward(Image convIn);
    void setOutputLayer(Layer out);
};

