#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <math.h>
#include "device_launch_parameters.h"
#include "fb_multi_channel_Impl.cuh"

__global__ void multiply_sum(cufftComplex* signal, cufftComplex* resultVec, float* filterTaps,
                            unsigned step, unsigned filterLen, unsigned channelCount, unsigned fftSize)
{
    unsigned sub_batch_count = filterLen / blockDim.x;
    unsigned sub_batch_index = blockIdx.x % sub_batch_count;
    unsigned h_index = sub_batch_index * blockDim.x + threadIdx.x;
    unsigned f_index = h_index % fftSize;
    unsigned batch_index = blockIdx.x / sub_batch_count;
    unsigned index = (batch_index * step + h_index) * channelCount;
    unsigned res_index = batch_index * fftSize + f_index;

    float tap = filterTaps[h_index];
    for (unsigned i = 0; i < channelCount; ++i)
    {
        unsigned new_res_index = channelCount * res_index + i;
        atomicAdd(&(resultVec[new_res_index].x), tap * signal[index + i].x);
        atomicAdd(&(resultVec[new_res_index].y), tap * signal[index + i].y);
    }
}



int executeImpl(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen,
                unsigned fftSize, unsigned step, unsigned channelCount, float* result,
                unsigned long resultLen, unsigned threads_per_block)
{
    if (threads_per_block > fftSize){
        threads_per_block = fftSize;
    }

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
    int * nx = new int(fftSize);
    int idist = channelCount * fftSize;
    int odist = fftSize;
    int istride = channelCount, ostride = 1;
    int *inembed = new int(resultLen);
    int *onembed = new int(resultLen);
    cufftStatus = cufftPlanMany(&plan, 1, nx, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, fftCount);
    if (cufftStatus != CUFFT_SUCCESS) {
        fprintf(stderr, "cufftPlanMany failed. Error code %d!\n", cufftStatus);
        return cudaErrorUnknown;
    }

    float* dev_inSignal;
    float* dev_filterTaps;
    cufftComplex* dev_result;
    cufftComplex* dev_tensor;

    unsigned num_Blocks;
    num_Blocks = fftCount * ceil((double)filterLen / threads_per_block);
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

    cudaStatus = cudaMalloc((float**)&dev_tensor, resultLen * sizeof(float));
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

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cufftComplex* dev_inComplexSignal = reinterpret_cast<cufftComplex*>(dev_inSignal);

    multiply_sum << <num_Blocks, threads_per_block >> > (dev_inComplexSignal, dev_result,
                                                          dev_filterTaps, step, filterLen,
                                                          channelCount, fftSize);

    for (int i = 0; i < channelCount; ++i)
    {
        cufftStatus = cufftExecC2C(plan, dev_result + i, dev_tensor + i*(fftSize*fftCount), CUFFT_FORWARD);
        if (cufftStatus != CUFFT_SUCCESS) {
            fprintf(stderr, "cufftExecC2C failed. Error code %d!\n", cufftStatus);
            return cudaErrorUnknown;
        }
    }
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("cuda execution time (without cufftPlan and memory operations): %f ms\n", milliseconds);

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

    cudaStatus = cudaMemcpy(result, reinterpret_cast<float*>(dev_tensor), resultLen * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return cudaStatus;
    }

    cudaFree(dev_inSignal);
    cudaFree(dev_filterTaps);
    cudaFree(dev_result);
    cudaFree(dev_tensor);

    return cudaStatus;
}


__inline__ __device__ cufftComplex operator + (cufftComplex const& a, cufftComplex const& b) {
    cufftComplex c;
    c.x = a.x + b.x;
    c.y = a.y + b.y;
    return c;
}