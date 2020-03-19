#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include "device_launch_parameters.h"
#include "../include/fb_multi_channel_Impl.cuh"

__global__ void mupltiply_sum(cufftComplex* signal, cufftComplex* resultVec, float* filterTaps, int k, int step, int filterLen, int channelCount)
{
    int index = (blockIdx.x * step + threadIdx.x)*channelCount;
    int res_index = blockIdx.x * blockDim.x + threadIdx.x;
    cufftComplex result;
    result.x = 0;
    result.y = 0;

    for (int i = 0; i < k; ++i)
    {
        int sig_index = i * blockDim.x * channelCount + index;
        result.x += filterTaps[i * blockDim.x + threadIdx.x] * signal[sig_index].x;
        result.y += filterTaps[i * blockDim.x + threadIdx.x] * signal[sig_index].y;
    }

    resultVec[res_index].x = result.x;
    resultVec[res_index].y = result.y;
}


int executeImpl(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen,
                    unsigned fftSize, unsigned step, unsigned channelCount, float* result, unsigned long resultLen)
{
    unsigned fftCount = ((signalLen / 2 - filterLen) / step) + 1;
    cufftHandle plan;
    cufftResult cufftStatus;
    cufftStatus = cufftPlan1d(&plan, fftSize, CUFFT_C2C, fftCount * channelCount);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftPlan1d failed. Error code %d!\n", cufftStatus);
        return cudaErrorUnknown;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start);

    float* dev_inSignal;
    float* dev_filterTaps;
    cufftComplex* dev_result;
    
    cudaError_t cudaStatus;
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

    for (int channelIndex = 0; channelIndex < channelCount; channelIndex++) {
        mupltiply_sum << <fftCount, fftSize >> > (dev_inComplexSignal + channelIndex, dev_result + fftCount*fftSize*channelIndex,
            dev_filterTaps, filterLen / fftSize, step, filterLen, channelCount);
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
    cudaFree(dev_result);


    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda execution time (without cufftPlan1d): %f ms\n", milliseconds);

    return cudaStatus;
}


__inline__ __device__ cufftComplex operator + (cufftComplex const& a, cufftComplex const& b) {
    cufftComplex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}