#include <stdio.h>
#include "filterbank.hpp"

void readMetadataFromFile(const char* fileName, unsigned* result);
void readVectorFromFile(const char* fileName, float* result, unsigned long len);
void writeResultToFile(const char* fileName, float* result, unsigned long len);

int main() {
    unsigned metadata[5];
    readMetadataFromFile("../python/files/metadata", metadata);
    const unsigned channelCount = metadata[0];
    const unsigned signalLen = metadata[1];
    const unsigned filterLen = metadata[2];
    const unsigned fftSize = metadata[3];
    const unsigned step = metadata[4];

    unsigned newSignalLen;
    if (signalLen % filterLen == 0){
        newSignalLen = signalLen;
    }
    else{
        newSignalLen = signalLen + filterLen - signalLen % filterLen;
    }

    unsigned fftCount = ((newSignalLen - filterLen) / (step)) + 1;
    const unsigned long resultLen = 2 * fftSize * fftCount * channelCount;
    float* result = new float[resultLen];

    printf("C = %d, N = %d, T = %d, F = %d, K = %d, fft count = %d\n", channelCount, signalLen, filterLen, fftSize, step, fftCount);

    float inSignal[2*signalLen * channelCount];
    float filterTaps[filterLen];

    readVectorFromFile("../python/files/signal", inSignal, 2*signalLen * channelCount);
    readVectorFromFile("../python/files/taps", filterTaps, filterLen);

    filterbank fb(signalLen, channelCount, fftSize, step, filterLen, filterTaps);
    fb.execute(inSignal, result);

    writeResultToFile("../python/files/result", result, resultLen);

    return 0;
}

void readVectorFromFile(const char* fileName, float* result, unsigned long len) {
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

void readMetadataFromFile(const char* fileName, unsigned* result) {
    FILE* file;
    file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Error reading file!!\n");
        return;
    }
    for (int m = 0; m < 5; ++m) {
        fscanf(file, "%d ", &result[m]);
    }
    fclose(file);
}

void writeResultToFile(const char* fileName, float* result, unsigned long len) {
    FILE* file;
    file = fopen(fileName, "w");

    for (int i = 0; i < len; ++i)
    {
        fprintf(file, "%f ", result[i]);
    }

    fclose(file);
}
