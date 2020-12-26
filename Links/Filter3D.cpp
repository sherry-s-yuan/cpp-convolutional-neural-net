#include "Filter3D.h"

Filter3D::Filter3D(int ic, int oc, int fs, string filterType) {
	inChannel = ic;
	outChannel = oc;
	filterSize = fs;
	filterChannel = inChannel - outChannel + 1;
    FilterFactory filterFactory = FilterFactory();
    filter = filterFactory.construct3D(filterType, filterChannel, fs, fs, 1);
}
Filter3D::Filter3D() {
    FilterFactory filterFactory = FilterFactory();
    filter = filterFactory.construct3D("gaussian", 1, 5, 5, 1);
    filterSize = 5;
}

Image Filter3D::forward(Image convIn, int skip) {
    int filterHeight = filterSize;
    int filterWidth = filterSize;

    int height = (convIn[0].size() - filterHeight) / skip + 1;
    int width = (convIn[0][0].size() - filterWidth) / skip + 1;

    Image fOutput(outChannel, Matrix(height, Array(width)));

    int d, i, j, h, w, c;

    // for each channel (3) apply the filter to each local receptive field
    for (d = 0; d < outChannel; d++) {
        // loop through each index of image
        for (i = 0; i < height; i++) {
            for (j = 0; j < width; j++) {
                // apply filter
                int imgCnl = d;
                int imgRow = i * skip;
                int imgCol = j * skip;
                for (c = imgCnl; c < imgCnl + filterChannel; c++) {
                    for (h = imgRow; h < imgRow + filterHeight; h++) {
                        for (w = imgCol; w < imgCol + filterWidth; w++) {
                            fOutput[d][i][j] += filter[c - imgCnl][h - imgRow][w - imgCol] * convIn[c][h][w];
                        }
                    }
                }
            }
        }
    }
    return fOutput;
}

Image Filter3D::backward(Image dConvOut, Image convIn, int skip) {
    int filterHeight = filterSize;
    int filterWidth = filterSize;
    int filterChannel = inChannel - outChannel + 1;

    int convOutHeight = dConvOut[0].size();
    int convOutWidth = dConvOut[0][0].size();
    int convOutChannel = dConvOut.size();

    Image df(filterChannel, Matrix(filterHeight, Array(filterWidth)));

    int d, i, j, h, w, c;

    // for each channel (3) apply the filter to each local receptive field
    for (d = 0; d < convOutChannel; d++) {
        // loop through each index of image
        for (i = 0; i < convOutHeight; i++) {
            for (j = 0; j < convOutWidth; j++) {
                // apply filter
                int imgCnl = d;
                int imgRow = i * skip;
                int imgCol = j * skip;
                for (c = imgCnl; c < imgCnl + filterChannel; c++) {
                    for (h = imgRow; h < imgRow + filterHeight; h++) {
                        for (w = imgCol; w < imgCol + filterWidth; w++) {
                            df[c - imgCnl][h - imgRow][w - imgCol] += convIn[c][h][w] * dConvOut[d][i][j];
                        }
                    }
                }
            }
        }
    }
    dFilter = df;
    return df;
}
