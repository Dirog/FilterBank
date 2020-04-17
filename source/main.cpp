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
    float filterTaps[filterLen];

    float* dev_result1;
    float* dev_result2;
    float* dev_result3;
    float* result = new float[2 * resultLen];

    float* inSignal1 = new float[2 * total_signalLen];
    float* inSignal2 = new float[2 * total_signalLen];
    float* inSignal3 = new float[2 * total_signalLen];
    float* dev_inSignal1;
    float* dev_inSignal2;
    float* dev_inSignal3;


    cudaError_t cudaStatus;
    cudaStatus = cudaMallocManaged((float**)&dev_result1, 2 * resultLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return -1;
    }

    cudaStatus = cudaMallocManaged((float**)&dev_result2, 2 * resultLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return -1;
    }

    cudaStatus = cudaMallocManaged((float**)&dev_result3, 2 * resultLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return -1;
    }

    cudaStatus = cudaMallocManaged((float**)&dev_inSignal1, 2 * total_signalLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return -1;
    }
    cudaStatus = cudaMallocManaged((float**)&dev_inSignal2, 2 * total_signalLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return -1;
    }

    cudaStatus = cudaMallocManaged((float**)&dev_inSignal3, 2 * total_signalLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return -1;
    }

    cudaStatus = cudaMalloc((float**)&dev_filterTaps, filterLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return -1;
    }

    printf("C = %d, N = %d, T = %d, F = %d, K = %d, fft count = %d\n", channelCount, signalLen, filterLen, fftSize, step, fftCount);

    readVectorFromFile("../python/files/signal1", inSignal1, 2 * total_signalLen);
    cudaStatus = cudaMemcpy(dev_inSignal1, inSignal1, 2 * total_signalLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!!\n");
        return -1;
    }

    readVectorFromFile("../python/files/signal2", inSignal2, 2 * total_signalLen);
    cudaStatus = cudaMemcpy(dev_inSignal2, inSignal2, 2 * total_signalLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!!\n");
        return -1;
    }

    readVectorFromFile("../python/files/signal3", inSignal3, 2 * total_signalLen);
    cudaStatus = cudaMemcpy(dev_inSignal3, inSignal3, 2 * total_signalLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!!\n");
        return -1;
    }


    readVectorFromFile("../python/files/taps", filterTaps, filterLen);
    cudaStatus = cudaMemcpy(dev_filterTaps, filterTaps, filterLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!!\n");
        return -1;
    }

    int threadsPerBlock = 128;
    Filterbank fb(signalLen, channelCount, fftSize, step, filterLen, dev_filterTaps, threadsPerBlock);


    fb.execute(dev_inSignal1, dev_result1);
    cudaStatus = cudaMemcpy(result, dev_result1, 2 * resultLen * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! res1\n");
        return -1;
    }
    writeVectorToFile("../python/files/result1", result, 2 * resultLen);

    
    fb.execute(dev_inSignal2, dev_result2);
    cudaStatus = cudaMemcpy(result, dev_result2, 2 * resultLen * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return -1;
    }
    writeVectorToFile("../python/files/result2", result, 2 * resultLen);


    fb.execute(dev_inSignal3, dev_result3);
    cudaStatus = cudaMemcpy(result, dev_result3, 2 * resultLen * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return -1;
    }
    writeVectorToFile("../python/files/result3", result, 2 * resultLen);



    cudaFree(dev_result1);
    cudaFree(dev_result2);
    cudaFree(dev_result3);
    cudaFree(dev_inSignal1);
    cudaFree(dev_inSignal2);
    cudaFree(dev_inSignal3);
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

int readVectorFromFile(const char* filePath, float* result, unsigned len){
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

int writeVectorToFile(const char* filePath, float* vector, unsigned long len){
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