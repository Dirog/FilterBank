#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "device_launch_parameters.h"



cudaError_t execute(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen, unsigned fftSize, unsigned step, float* result, unsigned resultLen);
void readVectorFromFile(char* fileName, float* result, int len);
void writeResultToFile(char* fileName, float* result, int x, int y);

__global__ void multiplyKernel(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen, unsigned fftSize, float* result, unsigned resultLen) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < filterLen)
    {
        result[2 * i] = inSignal[2 * i] * filterTaps[i];
        result[2 * i + 1] = inSignal[2 * i + 1] * filterTaps[i];
    }


}


int main() {
    const int signalLen = 16 * 2;
    const int filterLen = 8;
    const int fftSize = filterLen;
    const int step = 4;
    const int fftCount = ((signalLen / 2 - filterLen / 2) / (step)) - 1;
    const int resultLen = 2 * fftSize * fftCount;
    float result[resultLen];

    float inSignal[signalLen];
    float filterTaps[filterLen];

    readVectorFromFile("sine", inSignal, signalLen);
    readVectorFromFile("taps", filterTaps, filterLen);

    execute(inSignal, signalLen, filterTaps, filterLen, fftSize, step, result, resultLen);

    writeResultToFile("result", result, 2 * fftSize, fftCount);

    /*for (int i = 0; i < resultLen / 2; i++) {
        printf("%f", result[2 * i]);
        if (result[2 * i + 1] >= 0)
            printf(" + %fi\n", result[2 * i + 1]);
        else
            printf(" %fi\n", result[2 * i + 1]);
    }*/

    return 0;
}

cudaError_t execute(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen, unsigned fftSize, unsigned step, float* result, unsigned resultLen)
{
    float* dev_inSignal = 0;
    float* dev_filterTaps = 0;
    int fftCount = ((signalLen / 2 - filterLen / 2) / (step)) - 1;
    float* dev_result = 0;
    float* dev_mlpResult = 0;
    int mlpResultLen = filterLen;
    cudaError_t cudaStatus;
    cufftResult cufftStatus;

    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_inSignal, signalLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_result, resultLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_mlpResult, mlpResultLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }


    cudaStatus = cudaMalloc((void**)&dev_filterTaps, filterLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inSignal, inSignal, signalLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    cudaStatus = cudaMemcpy(dev_filterTaps, filterTaps, filterLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }


    cuffthandle plan;
    cufftstatus = cufftplan1d(&plan, 2 * fftsize, cufft_c2c, 1);
    if (cufftstatus != cudasuccess) {
        fprintf(stderr, "cufftplan failed!");
        goto error;
    }

    for (int i = 0; i < fftCount; i++)
    {
        multiplyKernel<<<256, 256>>> (dev_inSignal, signalLen, dev_filterTaps, filterLen, fftSize, dev_mlpResult, mlpResultLen);

        cufftStatus = cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(dev_mlpResult),
            reinterpret_cast<cufftComplex*>(dev_result + i * 2 * fftSize),
            CUFFT_FORWARD);
        if (cufftStatus != cudaSuccess) {
            fprintf(stderr, "cufftExec failed!");
            goto Error;
        }
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        goto Error;
    }


    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        goto Error;
    }

    cudaStatus = cudaMemcpy(result, reinterpret_cast<float*>(dev_result), resultLen * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(inSignal, dev_inSignal, signalLen * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }

Error:
    cudaFree(dev_inSignal);
    cudaFree(dev_filterTaps);

    return cudaStatus;
}


void readVectorFromFile(char* fileName, float* result, int len) {
    FILE* signal_file;
    signal_file = fopen(fileName, "r");
    if (signal_file == NULL) {
        printf("Error reading file!");
        return;
    }
    for (int m = 0; m < len; ++m) {
        fscanf(signal_file, "%f ", &result[m]);
    }
    fclose(signal_file);
}

void writeResultToFile(char* fileName, float* result, int x, int y) {
    FILE* file;
    file = fopen(fileName, "w");

    int  n = 0;
    for (int l = 0; l < y; ++l) {
        for (int i = 0; i < x; ++i) {
            fprintf(file, "%f ", result[n]);
            n++;
        }
        fprintf(file, "\n");
    }
    fclose(file);
}