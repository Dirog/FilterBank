#include "cuda_runtime.h"
#include "device_launch_parameters.h"


__global__ void multiplyKernel(float * inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen, float * result){
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    int j = 0;
    if (i < signalLen) {
        result[2*i] = inSignal[2*i] * filterTaps[j]; //Re
        result[2*i + 1] = inSignal[2*i + 1] * filterTaps[j]; //Im
        j++;
        if (j >= filterLen) {
            j = 0;
        }
    }
}

cudaError_t multiplyWithCuda(float * inSignal, unsigned signalLen, float* dev_filterTaps, unsigned filterLen, float * result)
{
    float * dev_inSignal; 
    float *dev_result;
    unsigned resultLen = signalLen;
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

    cudaStatus = cudaMalloc((void**)&dev_result, resultLen * sizeof(float));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!");
        goto Error;
    }

    cudaStatus = cudaMemcpy(dev_inSignal, inSignal, signalLen * sizeof(float), cudaMemcpyHostToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!");
        goto Error;
    }



    // int N = 256;
    // dim3 threadsPerBlock(N, N);
    // dim3 numBlocks;    
    // multiplyKernel<<<numBlocks, threadsPerBlock>>>(dev_inSignal, signalLen, dev_filterTaps, filterLen, dev_result);


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

//     cudaStatus = cudaMemcpy(result, dev_result, resultLen * sizeof(float), cudaMemcpyDeviceToHost);
//     if (cudaStatus != cudaSuccess) {
//         fprintf(stderr, "cudaMemcpy failed!");
//         goto Error;
//     }

// Error:
//     cudaFree(dev_inSignal);
//     cudaFree(dev_result);
    
    return cudaStatus;
}
