#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "device_launch_parameters.h"




cudaError_t multiplyAndAverageWithCuda(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen, unsigned fftSize, float* result);

__global__ void multiplyKernel(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen, unsigned fftSize, float* result, unsigned resultLen) {
    
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < signalLen)
    {
        int j = i % filterLen;
        result[2 * i] = inSignal[2 * i] * filterTaps[j];
        result[2 * i + 1] = inSignal[2 * i + 1] * filterTaps[j];
    }


}


int main() {
    const int arrSize = 2 * 16;
    const int hSize = 8;
    const int fftSize = 8;
    const int resultLen = fftSize * (arrSize / hSize);
    float result[resultLen];
    float inSignal[arrSize] = { 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13, 13, 14, 14, 15, 15, 16, 16 };
    float h[hSize] = { 1, 2, 3, 4, 5, 6, 7, 8 };

    multiplyAndAverageWithCuda(inSignal, arrSize, h, hSize, fftSize, result);
   
    return 0;
}

cudaError_t multiplyAndAverageWithCuda(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen, unsigned fftSize, float* result)
{
    float* dev_inSignal = 0;
    float* dev_result = 0;
    float* dev_filterTaps = 0;
    const int resultLen = fftSize * (signalLen / filterLen);
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



    multiplyKernel <<<256, 1024>>> (dev_inSignal, signalLen, dev_filterTaps, filterLen, fftSize, dev_result, resultLen);


    cufftHandle plan;
    cufftStatus = cufftPlan1d(&plan, fftSize, CUFFT_C2C, signalLen / fftSize);
    if (cufftStatus != cudaSuccess) {
        fprintf(stderr, "cufftPlan failed!");
        goto Error;
    }

    cufftStatus = cufftExecC2C(plan, reinterpret_cast<cufftComplex*>(dev_result),
                    reinterpret_cast<cufftComplex*>(dev_result),
                    CUFFT_FORWARD);
    if (cufftStatus != cudaSuccess) {
        fprintf(stderr, "cufftExec failed!");
        goto Error;
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

    cudaStatus = cudaMemcpy(result, dev_result, resultLen * sizeof(float), cudaMemcpyDeviceToHost);
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
