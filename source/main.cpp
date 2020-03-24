#include <stdio.h>
#include <vector>
#include <fstream>
#include <iostream>
#include "filterbank.hpp"

using namespace std;

int readVectorFromFile(const char * filePath, float * result, unsigned len);
int writeVectorToFile(const char * filePath, float * vector, unsigned len);
void readMetadataFromFile(const char* fileName, unsigned* result);


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

    float * inSignal = new float[2*signalLen * channelCount];
    printf("!\n");
    float filterTaps[filterLen];


    readVectorFromFile("../python/files/signal", inSignal, 2*signalLen * channelCount);
    readVectorFromFile("../python/files/taps", filterTaps, filterLen);

    
    filterbank fb(signalLen, channelCount, fftSize, step, filterLen, filterTaps);
    fb.execute(inSignal, result);

    writeVectorToFile("../python/files/result", result, resultLen);

    return 0;
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

int readVectorFromFile(const char * filePath, float * result, unsigned len){
    ifstream rf(filePath, ios::out | ios::binary);
    if(!rf) {
        cout << "Cannot open file!" << endl;
        return -1;
    }

    for(int i = 0; i < len; i++)
        rf.read((char *) &result[i], sizeof(float));

    rf.close();
    if(!rf.good()) {
        cout << "Error occurred at reading time!" << endl;
        return -1;
    }
    return 0;
}

int writeVectorToFile(const char * filePath, float * vector, unsigned len){
    ofstream wf(filePath, ios::out | ios::binary);
   if(!wf) {
      cout << "Cannot open file!" << endl;
      return -1;
   }

   for(int i = 0; i < len; i++)
      wf.write((char *) &vector[i], sizeof(float));

   wf.close();
   if(!wf.good()) {
      cout << "Error occurred at writing to file time!" << endl;
      return -1;
   }
   return 0;
}
