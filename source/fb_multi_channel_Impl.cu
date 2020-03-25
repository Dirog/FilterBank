#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <vector>
#include <math.h>
#include "device_launch_parameters.h"
#include "fb_multi_channel_Impl.cuh"

__global__ void mupltiply_sum(cufftComplex* signal, cufftComplex* resultVec, float* filterTaps, unsigned k,
                                unsigned step, unsigned filterLen, unsigned channelCount, unsigned fftSize, unsigned fftCount)
{
    unsigned sub_batch_size = fftSize / blockDim.x;
    unsigned sub_batch_index = blockIdx.x % sub_batch_size;
    unsigned f_index = sub_batch_index * sub_batch_size + threadIdx.x;
    unsigned batch_index = blockIdx.x / sub_batch_size;
    unsigned index = (batch_index * step + f_index) * channelCount;
    unsigned res_index = batch_index * fftSize + f_index;
    //printf("sub_batch_size %d, sub_batch_index %d, batch_index %d, f_index %d, index %d, res_index %d\n", sub_batch_size, sub_batch_index, batch_index, f_index, index, res_index);
    cufftComplex result;
    result.x = 0;
    result.y = 0;

    for (unsigned i = 0; i < k; ++i)
    {
        unsigned sig_index = i * fftSize * channelCount + index;
        unsigned h_index = i * fftSize + f_index;
        result.x += filterTaps[h_index] * signal[sig_index].x;
        result.y += filterTaps[h_index] * signal[sig_index].y;
    }

    resultVec[res_index].x = result.x;
    resultVec[res_index].y = result.y;
}



int executeImpl(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen,
                unsigned fftSize, unsigned step, unsigned channelCount, float* result, unsigned long resultLen)
{
    unsigned zerosToPad;
    if (signalLen % filterLen == 0){
        zerosToPad = 0;
    }
    else{
        zerosToPad = filterLen - signalLen % filterLen;
    }
    printf("Zeros to pad: %d\n", zerosToPad);
    unsigned newSignalLen = signalLen + zerosToPad;
    unsigned fftCount = ((newSignalLen - filterLen) / step) + 1;

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

    unsigned threads_per_block;
    unsigned num_Blocks;

    threads_per_block = 1024;
    num_Blocks = fftCount * ceil((double)fftSize / threads_per_block);
    printf("threads_per_block %d, num_Blocks %d\n", threads_per_block, num_Blocks);
    
    cudaError_t cudaStatus;
    cudaStatus = cudaSetDevice(0);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        return cudaStatus;
    }

    
    cudaStatus = cudaMalloc((float**)&dev_inSignal, sizeof(float) * channelCount * 2 * newSignalLen);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! 1\n");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((float**)&dev_result, resultLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! 2\n");
        return cudaStatus;
    }

    cudaStatus = cudaMalloc((float**)&dev_filterTaps, filterLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed! 3\n");
        return cudaStatus;
    }

    float * zeros = new float[2 * zerosToPad * channelCount]();
    cudaStatus = cudaMemcpy(dev_inSignal, zeros, 2 * zerosToPad * channelCount * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! 4\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_inSignal + channelCount * 2 * zerosToPad, inSignal, 2 * signalLen * channelCount * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! 5\n");
        return cudaStatus;
    }
   
    cudaStatus = cudaMemcpy(dev_filterTaps, filterTaps, filterLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! 6\n");
        return cudaStatus;
    }


    cufftComplex* dev_inComplexSignal = reinterpret_cast<cufftComplex*>(dev_inSignal);

    for (int channelIndex = 0; channelIndex < channelCount; channelIndex++) {
        mupltiply_sum << <num_Blocks, threads_per_block >> > (dev_inComplexSignal + channelIndex, dev_result + fftCount*fftSize*channelIndex,
            dev_filterTaps, filterLen / fftSize, step, filterLen, channelCount, fftSize, fftCount);
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