#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include <cuda_runtime.h>
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
    const unsigned long resultLen = fftSize * fftCount * channelCount;
    const unsigned total_signalLen = signalLen * channelCount;
    float* dev_filterTaps;
    float* dev_inSignal;
    float* dev_result;
    float * inSignal = new float[2 * total_signalLen];
    float* result = new float[2 * resultLen];
    float filterTaps[filterLen];

    cudaError_t cudaStatus;
    cudaStatus = cudaMalloc((float**)&dev_result, 2 * resultLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return -1;
    }

    cudaStatus = cudaMalloc((float**)&dev_inSignal, 2 * total_signalLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return -1;
    }

    cudaStatus = cudaMalloc((float**)&dev_filterTaps, filterLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return -1;
    }


    printf("C = %d, N = %d, T = %d, F = %d, K = %d, fft count = %d\n",
        channelCount, signalLen, filterLen, fftSize, step, fftCount);

    int signalStatus;
    int tapsStatus;
    signalStatus = readVectorFromFile("../python/files/signal", inSignal, 2 * total_signalLen);
    tapsStatus = readVectorFromFile("../python/files/taps", filterTaps, filterLen);
    if (signalStatus != 0 || tapsStatus != 0){
        return -1;
    }

    cudaStatus = cudaMemcpy(dev_filterTaps, filterTaps, filterLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!!\n");
        return -1;
    }

    int threadsPerBlock = 128;
    Filterbank fb(signalLen, channelCount, fftSize, step, filterLen, dev_filterTaps, threadsPerBlock);


    cudaStatus = cudaMemcpy(dev_inSignal, inSignal, 2 * total_signalLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!!\n");
        return -1;
    }

    int status;
    status = fb.execute(dev_inSignal, dev_result);

    cudaStatus = cudaMemcpy(result, dev_result, 2 * resultLen * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return -1;
    }


    int writeStatus;
    writeStatus = writeVectorToFile("../python/files/result", result, 2 * resultLen);

    if (status == 0 && writeStatus == 0){
        printf("Success\n");
    }
    else{
        return -1;
    }

    cudaFree(dev_result);
    cudaFree(dev_inSignal);
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