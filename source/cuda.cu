#include <stdio.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"



cudaError_t multiplyAndAverageWithCuda(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen, unsigned fftSize, float* result);

__global__ void multiplyKernel(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen, float* result) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < signalLen) {
        int j = i % filterLen;
        result[2 * i] = inSignal[2 * i] * filterTaps[j]; //Re
        result[2 * i + 1] = inSignal[2 * i + 1] * filterTaps[j]; //Im
    }
}

__global__ void averageKernel(float * arr, unsigned arrLen, unsigned windowSize, unsigned size, float * result, unsigned resultLen) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = blockDim.y * blockIdx.y + threadIdx.y;

    int k = 0.5 * arrLen / size;
    float sumRe = 0;
    float sumIm = 0;
    if (i < size) {
        if (j < k) {
            sumRe += arr[(i + j * size)*2];
            sumIm += arr[(i + j * size)*2 + 1];
        }
        result[2 * i] = sumRe;
        result[2 * i + 1] = sumIm;
    }


}

int main() {
    const int arrSize = 16;
    const int hSize = 4;
    const int fftSize = 2;
    const int resultLen = fftSize * (arrSize / hSize);
    float result[resultLen];
    float inSignal[arrSize] = { 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8 };
    float h[hSize] = { 1, 2, 3, 4 };

    multiplyAndAverageWithCuda(inSignal, arrSize, h, hSize, fftSize, result);

    return 0;
}

cudaError_t multiplyAndAverageWithCuda(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen, unsigned fftSize, float* result)
{
    float* dev_inSignal = 0;
    float* dev_multiplicationResult = 0;
    float* dev_result = 0;
    float* dev_filterTaps = 0;
    unsigned multiplicationResultLen = signalLen;
    unsigned resultLen = 2 * fftSize * (signalLen / filterLen);
    cudaError_t cudaStatus;

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

    cudaStatus = cudaMalloc((void**)&dev_multiplicationResult, multiplicationResultLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMalloc((void**)&dev_result, resultLen * sizeof(float));
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



    int N = 16;
    dim3 threadsPerBlock(N, N);
    dim3 numBlocks;    
    multiplyKernel<<<numBlocks, threadsPerBlock>>>(dev_inSignal, signalLen, dev_filterTaps, filterLen, dev_result);
    //averageKernel <<<numBlocks, threadsPerBlock >>> (dev_multiplicationResult, multiplicationResultLen, filterLen, fftSize, result, resultLen);




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

         cudaStatus = cudaMemcpy(result, dev_result, resultLen * sizeof(float), cudaMemcpyDeviceToHost);
         if (cudaStatus != cudaSuccess) {
             fprintf(stderr, "cudaMemcpy failed!");
             goto Error;
         }

     Error:
         cudaFree(dev_inSignal);
         cudaFree(dev_multiplicationResult);
         cudaFree(dev_filterTaps);

    return cudaStatus;
}
