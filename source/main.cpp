#include <stdio.h>
#include "../include/main.hpp"
#include "../include/filterbank.hpp"

using namespace std;

int main() {
    const unsigned channelCount = 3;
    const unsigned signalLen = 1024 * 256 * 2;
    const unsigned filterLen = 1024;
    const unsigned fftSize = filterLen / 128;
    const unsigned step = 128;

    unsigned fftCount = ((signalLen / 2 - filterLen) / (step)) + 1;
    const unsigned long resultLen = 2 * fftSize * fftCount * channelCount;
    float* result = new float[resultLen];
    printf("C = %d, N = %d, T = %d, F = %d, K = %d, fft count = %d\n", channelCount, signalLen, filterLen, fftSize, step, fftCount);

    float inSignal[signalLen * channelCount];
    float filterTaps[filterLen];

    readVectorFromFile("./python/files/signal", inSignal, signalLen * channelCount);
    readVectorFromFile("./python/files/taps", filterTaps, filterLen);

    filterbank fb(signalLen, channelCount, fftSize, step, filterLen, filterTaps);
    fb.execute(inSignal, result);


    writeResultToFile("./python/files/result", result, resultLen);

    return 0;
}

void readVectorFromFile(const char* fileName, float* result, unsigned long len) {
    //printf("Reading from file! %d\n");
    FILE* file;
    file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Error reading file!!\n");
        return;
    }
    for (int m = 0; m < len; ++m) {
        fscanf(file, "%f ", &result[m]);
    }
    fclose(file);
}

void writeResultToFile(const char* fileName, float* result, unsigned long len) {
    //printf("Writing result to file!\n");
    FILE* file;
    file = fopen(fileName, "w");

    for (int i = 0; i < len; ++i)
    {
        fprintf(file, "%f ", result[i]);
    }

    fclose(file);
}
