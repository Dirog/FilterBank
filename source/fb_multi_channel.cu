#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "device_launch_parameters.h"
#include <iostream>

#define DIM 1024

cudaError_t execute(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen,
    unsigned fftSize, unsigned step, unsigned channelCount, float* result, unsigned resultLen);
void readVectorFromFile(const char* fileName, float* result, int len);
void writeResultToFile(const char* fileName, float* result, int len);
__device__ cufftComplex operator + (cufftComplex const& a, cufftComplex const& b);


__global__ void multiply(cufftComplex* signalVec, float* filterTaps, cufftComplex* vecOut, int size, int step, int channelCount, int stepIndx)
{
    unsigned int i = threadIdx.x + blockDim.x * blockIdx.x;
    if (i < size / step) {
        int indx = step * i + stepIndx;
        vecOut[i].x = signalVec[indx * channelCount].x * filterTaps[indx];
        vecOut[i].y = signalVec[indx * channelCount].y * filterTaps[indx];
    }
}

__global__ void reductionSum(cufftComplex* vect, cufftComplex* vecOut, int size)
{
    __shared__ cufftComplex block[DIM];
    unsigned int globalIndex = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned int i = threadIdx.x;

    if (globalIndex < size) {
        block[i] = vect[globalIndex];
    }
    else {
        block[i].x = 0;
        block[i].y = 0;
    }

    __syncthreads();

    for (unsigned int j = blockDim.x / 2; j > 0; j >>= 1)
    {
        if (i < j) {
            block[i] = block[i] + block[i + j];
        }

        __syncthreads();
    }

    if (i == 0) {
        vecOut[blockIdx.x] = block[0];
    }

}

int main() {
    const int channelCount = 3;
    const int signalLen = 1024 * 8 * 2;
    const int filterLen = 128;
    const int fftSize = filterLen / 16;
    const int step = 32;
    int fftCount = ((signalLen / 2 - filterLen) / (step)) + 1;
    const int resultLen = 2 * fftSize * fftCount * channelCount;
    float* result = new float[resultLen];
    //printf("C = %d, N = %d, T = %d, F = %d, K = %d, fft count = %d\n", channelCount, signalLen, filterLen, fftSize, step, fftCount);

    float inSignal[signalLen * channelCount];
    float filterTaps[filterLen];

    readVectorFromFile("../python/files/signal", inSignal, signalLen * channelCount);
    readVectorFromFile("../python/files/taps", filterTaps, filterLen);

    cudaError_t cudaStatus;
    cudaStatus = execute(inSignal, signalLen, filterTaps, filterLen, fftSize, step, channelCount, result, resultLen);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Execution failed\n");
        return -1;
    }

    writeResultToFile("../python/files/result", result, resultLen);

    return 0;
}

cudaError_t execute(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen,
    unsigned fftSize, unsigned step, unsigned channelCount, float* result, unsigned resultLen)
{
    int threadsPerBlock = DIM;

    float* dev_inSignal;
    float* dev_filterTaps;
    cufftComplex* dev_result;
    cufftComplex* dev_subVec;
    cufftComplex* dev_vecOut;

    int fftCount = ((signalLen / 2 - filterLen) / step) + 1;
    int subVecSize = filterLen / fftSize;
    int total_fftSize = fftSize * fftCount;

    cudaError_t cudaStatus;
    cufftResult cufftStatus;


    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((float**)&dev_inSignal, signalLen * channelCount * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMallocManaged((float**)&dev_result, resultLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((float**)&dev_filterTaps, filterLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((float**)&dev_subVec, filterLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMallocManaged((float**)&dev_vecOut, subVecSize * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_inSignal, inSignal, signalLen * channelCount * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_filterTaps, filterTaps, filterLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return cudaStatus;
    }

    cufftComplex* dev_inComplexSignal = reinterpret_cast<cufftComplex*>(dev_inSignal);
    int numInputElements = subVecSize;
    int numOutputElements;

    for (int channelIndx = 0; channelIndx < channelCount; channelIndx++) {
        for (int batchIndx = 0; batchIndx < fftCount; batchIndx++)
        {
            for (int stepIndx = 0; stepIndx < fftSize; stepIndx++) {
                multiply << <256, threadsPerBlock >> > (dev_inComplexSignal + batchIndx * step * channelCount + channelIndx, dev_filterTaps,
                    dev_subVec, filterLen, fftSize, channelCount, stepIndx);

                do
                {
                    numOutputElements = numInputElements / (threadsPerBlock);
                    if (numInputElements % (threadsPerBlock)) {
                        numOutputElements++;
                    }

                    reductionSum << < numOutputElements, threadsPerBlock >> > (dev_subVec, dev_vecOut, numInputElements);
                    numInputElements = numOutputElements;
                    if (numOutputElements > 1) {
                        reductionSum << < numOutputElements, threadsPerBlock >> > (dev_vecOut, dev_subVec, numInputElements);
                    }

                } while (numOutputElements > 1);

                cudaStatus = cudaDeviceSynchronize();
                if (cudaStatus != cudaSuccess) {
                    fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
                    return cudaStatus;
                }

                dev_result[stepIndx + batchIndx * fftSize + channelIndx * total_fftSize] = dev_vecOut[0];
                dev_vecOut[0].x = 0;
                dev_vecOut[0].y = 0;
                numInputElements = subVecSize;
            }
        }  
    }

    cufftHandle plan;
    cufftStatus = cufftPlan1d(&plan, fftSize, CUFFT_C2C, fftCount * channelCount);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftPlan1d failed. Error code %d!\n", cufftStatus);
        return cudaErrorUnknown;
    }

    cufftStatus = cufftExecC2C(plan, dev_result, dev_result, CUFFT_FORWARD);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftExecC2C failed. Error code %d!\n", cufftStatus);
        return cudaErrorUnknown;
    }

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed: %s\n", cudaGetErrorString(cudaStatus));
        return cudaStatus;
    }


    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(result, reinterpret_cast<float*>(dev_result), resultLen * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return cudaStatus;
    }

    cudaFree(dev_inSignal);
    cudaFree(dev_filterTaps);
    cudaFree(dev_subVec);
    cudaFree(dev_result);
    cudaFree(dev_vecOut);

    return cudaStatus;
}


__device__ cufftComplex operator + (cufftComplex const& a, cufftComplex const& b) {
    cufftComplex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}

void readVectorFromFile(const char* fileName, float* result, int len) {
    //printf("Reading from file! %d\n");
    FILE* file;
    file = fopen(fileName, "r");
    if (file == NULL) {
        printf("Error reading file!\n");
        return;
    }
    for (int m = 0; m < len; ++m) {
        fscanf(file, "%f ", &result[m]);
    }
    fclose(file);
}

void writeResultToFile(const char* fileName, float* result, int len) {
    //printf("Writing result to file!\n");
    FILE* file;
    file = fopen(fileName, "w");

    for (int i = 0; i < len; ++i)
    {
        fprintf(file, "%f ", result[i]);
    }

    fclose(file);
}