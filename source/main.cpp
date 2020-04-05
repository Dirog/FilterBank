#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include "filterbank.hpp"

int readVectorFromFile(const char * filePath, float * result, unsigned len);
int writeVectorToFile(const char * filePath, float * vector, unsigned long len);
int readMetadataFromFile(const char* fileName, unsigned* result);


int main() {
    int metadataStatus;
    unsigned metadata[5];
    metadataStatus = readMetadataFromFile("../python/files/metadata", metadata);
    if (metadataStatus != 0){
        return -1;
    }

    const unsigned channelCount = metadata[0];
    const unsigned signalLen = metadata[1];
    const unsigned filterLen = metadata[2];
    const unsigned fftSize = metadata[3];
    const unsigned step = metadata[4];

    unsigned fftCount = signalLen / step;
    const unsigned long resultLen = 2 * fftSize * fftCount * channelCount;
    float* result = new float[resultLen];


    printf("C = %d, N = %d, T = %d, F = %d, K = %d, fft count = %d\n",
        channelCount, signalLen, filterLen, fftSize, step, fftCount);

    float * inSignal = new float[2 * signalLen * channelCount];
    float filterTaps[filterLen];

    int signalStatus;
    int tapsStatus;
    signalStatus = readVectorFromFile("../python/files/signal", inSignal, 2 * signalLen * channelCount);
    tapsStatus = readVectorFromFile("../python/files/taps", filterTaps, filterLen);
    if (signalStatus != 0 || tapsStatus != 0){
        return -1;
    }

    int threadsPerBlock = 128;
    Filterbank fb(signalLen, channelCount, fftSize, step, filterLen, filterTaps, threadsPerBlock);

    int status;
    status = fb.execute(inSignal, result);

    int writeStatus;
    writeStatus = writeVectorToFile("../python/files/result", result, resultLen);

    if (status == 0 && writeStatus == 0){
        printf("Success\n");
    }
    else{
        return -1;
    }

    auto [x, y, z] = fb.getOutDim(); 

    printf("x = %d, y = %d, z = %d\n", x, y, z);
    return 0;
}

int readMetadataFromFile(const char* fileName, unsigned* result) {
    using namespace std;
    FILE* file;
    file = fopen(fileName, "r");
    if (file == NULL) {
        fprintf(stderr, "Error reading file!\n");
        return -1;
    }
    for (int m = 0; m < 5; ++m) {
        fscanf(file, "%d ", &result[m]);
    }
    fclose(file);
    return 0;
}

int readVectorFromFile(const char * filePath, float * result, unsigned len){
    using namespace std;
    ifstream rf(filePath, ios::out | ios::binary);
    if(!rf) {
        cerr << "Cannot open file!" << endl;
        return -1;
    }

    for(int i = 0; i < len; i++)
        rf.read((char *) &result[i], sizeof(float));

    rf.close();
    if(!rf.good()) {
        cerr << "Error occurred at reading time!" << endl;
        return -1;
    }
    return 0;
}

int writeVectorToFile(const char * filePath, float * vector, unsigned long len){
    using namespace std;
    ofstream wf(filePath, ios::out | ios::binary);
    if(!wf) {
      cerr << "Cannot open file!" << endl;
      return -1;
    }

    for(int i = 0; i < len; i++)
      wf.write((char *) &vector[i], sizeof(float));

    wf.close();
    if(!wf.good()) {
      cerr << "Error occurred at writing to file time!" << endl;
      return -1;
    }
    return 0;
}
