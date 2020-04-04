#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <cufftXt.h>
#include <math.h>
#include "device_launch_parameters.h"
#include "fb_multi_channel_Impl.cuh"
#include "filterbank.hpp"

__global__ void multiplyAndSum(cufftComplex* signal, cufftComplex* resultVec, float* filterTaps,
                            unsigned step, unsigned filterLen, unsigned channelCount, unsigned fftSize, unsigned totalSignalLen, unsigned sub_batch_count)
{
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
        unsigned signal_index = index + i;

        if(signal_index < totalSignalLen)
        {
            atomicAdd(&(resultVec[new_res_index].x), tap * signal[signal_index].x);
            atomicAdd(&(resultVec[new_res_index].y), tap * signal[signal_index].y);
        }
    }
}

__global__ void multiply(cufftComplex* tensor, cufftComplex* factors, unsigned fftCount, unsigned fftSize)
{
    unsigned x  = threadIdx.x + blockDim.x * blockIdx.x;
    unsigned y  = threadIdx.y + blockDim.y * blockIdx.y;
    unsigned k = x % fftCount;
    unsigned f = y % fftSize;

    unsigned tensor_index = x*fftSize + y;
    unsigned factor_index = k*fftSize + f;

    float a = tensor[tensor_index].x;
    float b = tensor[tensor_index].y;
    float c = factors[factor_index].x;
    float d = factors[factor_index].y;
    float re = a*c - b*d;
    float im = b*c + a*d;
    tensor[tensor_index].x = re;
    tensor[tensor_index].y = im;
}

int executeImpl(float* inSignal, unsigned signalLen, float* dev_filterTaps, unsigned filterLen,
                unsigned fftSize, unsigned step, unsigned channelCount, float* result,
                unsigned long resultLen, unsigned threads_per_block, cufftHandle plan, cufftComplex* dev_phaseFactors, cufftComplex* dev_history)
{
printf("resultLen = %d\n", resultLen / 2);

    if (threads_per_block > fftSize){
        threads_per_block = fftSize;
    }

    // unsigned zerosToPad = filterLen - 1;
    // printf("Zeros to pad: %d\n", zerosToPad);
    unsigned historyLen = filterLen - 1;
    unsigned newSignalLen = signalLen + historyLen;
    unsigned fftCount = signalLen / step;

    printf("new matrixSize  = %d\n", newSignalLen*channelCount);

    cufftResult cufftStatus;

    float* dev_inSignal;
    cufftComplex* dev_result;
    cufftComplex* dev_tensor;

    unsigned num_Blocks;
    num_Blocks = fftCount* ceil((double)filterLen / threads_per_block);
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
        fprintf(stderr, "cudaMalloc failed! 3\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_inSignal, dev_history, 2 * historyLen * channelCount * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! 4\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(dev_inSignal + 2 * historyLen * channelCount, inSignal, 2 * signalLen * channelCount * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed! 5\n");
        return cudaStatus;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cufftComplex* dev_inComplexSignal = reinterpret_cast<cufftComplex*>(dev_inSignal);

    multiplyAndSum << <num_Blocks, threads_per_block >> > (dev_inComplexSignal, dev_result,
                                                          dev_filterTaps, step, filterLen,
                                                          channelCount, fftSize, signalLen*channelCount, filterLen / threads_per_block);

    unsigned threads = (int)sqrt(threads_per_block);
    unsigned blocks = (int)sqrt(resultLen / 2 / threads_per_block);
    dim3 threadsInBlock (threads, threads);
    dim3 blocksPerGrid(blocks, blocks);

    for (int i = 0; i < channelCount; ++i)
    {
        cufftStatus = cufftExecC2C(plan, dev_result + i, dev_tensor + i*(fftSize*fftCount), CUFFT_FORWARD);
        if (cufftStatus != CUFFT_SUCCESS) {
            fprintf(stderr, "cufftExecC2C failed. Error code %d!\n", cufftStatus);
            return cudaErrorUnknown;
        }
    }
    
    //multiply <<<blocksPerGrid, threadsInBlock>>> (dev_tensor, dev_phaseFactors, fftCount, fftSize);

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


    unsigned endPos = signalLen * channelCount - (filterLen - 1) * channelCount;
    cudaStatus = cudaMemcpy(dev_history, dev_inSignal + endPos, (filterLen - 1) * channelCount * sizeof(cufftComplex), cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return cudaStatus;
    }

    cudaStatus = cudaMemcpy(result, reinterpret_cast<float*>(dev_tensor), resultLen * sizeof(float), cudaMemcpyDeviceToHost);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return cudaStatus;
    }

    cudaFree(dev_inSignal);
    cudaFree(dev_result);
    cudaFree(dev_tensor);

    return cudaStatus;
}