#include <cufft.h>
#include <stdio.h>
#include "filterbank.hpp"

__inline__ __device__ cufftComplex operator * (cufftComplex const& a, cufftComplex const& b) {
    cufftComplex c;
    c.x = a.x * b.x - a.y * b.y;
    c.y = a.x * b.y + a.y * b.x;
    return c;
}

__global__ void multiplyAndSum(cufftComplex* signal, cufftComplex* resultVec, cufftComplex* history,
    float* filterTaps, unsigned step, unsigned channelCount, unsigned fftSize, unsigned filterLen,
    unsigned totalSignalLen, unsigned totalHistoryLen, unsigned subBatchCount)
{
    unsigned sub_batch_index = blockIdx.x % subBatchCount;
    unsigned h_index = (sub_batch_index * blockDim.x + threadIdx.x);

    if (h_index < filterLen)
    {
        unsigned f_index = h_index % fftSize;
        unsigned batch_index = blockIdx.x / subBatchCount;
        unsigned index = (batch_index * step + h_index) * channelCount;
        unsigned res_index = batch_index * fftSize + f_index;


        float tap = filterTaps[h_index];
        for (unsigned i = 0; i < channelCount; ++i)
        {
            unsigned new_res_index = channelCount * res_index + i;
            unsigned signal_index = index + i;

            if (signal_index < totalHistoryLen)
            {
                atomicAdd(&(resultVec[new_res_index].x), tap * history[signal_index].x);
                atomicAdd(&(resultVec[new_res_index].y), tap * history[signal_index].y);
            }
            else if(signal_index < totalSignalLen + totalHistoryLen)
            {
                atomicAdd(&(resultVec[new_res_index].x), tap * signal[signal_index - totalHistoryLen].x);
                atomicAdd(&(resultVec[new_res_index].y), tap * signal[signal_index - totalHistoryLen].y);
            }
        }
    }
}

__global__ void multiply(cufftComplex* tensor, cufftComplex* factors, cufftComplex* initPhaseFactors, unsigned fftSize,
    unsigned fftCount, unsigned tensorlen) //Doesn't work
{
    unsigned index  = threadIdx.x + blockDim.x * blockIdx.x;
    if(index < tensorlen)
    {
        unsigned f = index % fftSize;
        unsigned factor_index = index % (fftCount * fftSize);
        tensor[index] = tensor[index] * factors[factor_index] * initPhaseFactors[f];
    }
}

__global__ void multiplyTest(cufftComplex* tensor, cufftComplex* factors, cufftComplex* initPhaseFactors, unsigned fftSize,
    unsigned fftCount, unsigned tensorlen) //Works
{
    for (int i = 0; i < tensorlen; ++i)
    {
        unsigned f = i % fftSize;
        unsigned factor_index = i % (fftCount * fftSize);
        tensor[i] = tensor[i] * factors[factor_index] * initPhaseFactors[f];
    }

}

__global__ void updateInitPhaseFactors(cufftComplex* initPhaseFactors, unsigned signalLen, unsigned fftSize)
{
    unsigned index  = threadIdx.x + blockDim.x * blockIdx.x;
    if (index < fftSize){
        unsigned f = index;

        double arg = -2 * M_PI * f * signalLen / fftSize;
        cufftComplex phase;
        phase.x = cos(arg);
        phase.y = sin(arg);
        initPhaseFactors[f] = initPhaseFactors[f] * phase;
    }
}

int executeImpl(float* dev_inSignal, unsigned signalLen, float* dev_filterTaps, unsigned filterLen, unsigned fftSize,
    unsigned step, unsigned channelCount, float* dev_result, unsigned long resultLen, unsigned threads_per_block,
    cufftHandle plan, cufftComplex* dev_phaseFactors, cufftComplex* dev_history, cufftComplex* dev_initPhaseFactors)
{
    unsigned historyLen = filterLen - 1;
    unsigned total_historyLen = historyLen * channelCount;
    unsigned total_signalLen = signalLen * channelCount;
    unsigned fftCount = signalLen / step;
    unsigned total_fftSize = fftCount * fftSize;
    cufftComplex* dev_tensor;

    unsigned num_Blocks;
    num_Blocks = fftCount * ceil((float)filterLen / threads_per_block);

    cudaError_t cudaStatus;
    cudaStatus = cudaMallocManaged((cufftComplex**)&dev_tensor, resultLen * sizeof(cufftComplex));
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMalloc failed!\n");
        return cudaStatus;
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    cufftComplex* dev_inComplexSignal = reinterpret_cast<cufftComplex*>(dev_inSignal);
    cufftComplex* dev_complexResult = reinterpret_cast<cufftComplex*>(dev_result);

    multiplyAndSum <<<num_Blocks, threads_per_block >>> (dev_inComplexSignal, dev_tensor, dev_history, dev_filterTaps,
        step, channelCount, fftSize, filterLen, total_signalLen, total_historyLen, ceil((float)filterLen / threads_per_block));

    cufftResult cufftStatus;
    for (int i = 0; i < channelCount; ++i)
    {
        cufftStatus = cufftExecC2C(plan, dev_tensor + i, dev_complexResult + i * total_fftSize, CUFFT_FORWARD);
        if (cufftStatus != CUFFT_SUCCESS) {
            fprintf(stderr, "cufftExecC2C failed. Error code %d!\n", cufftStatus);
            return cudaErrorUnknown;
        }
    }

    num_Blocks = ceil((float)resultLen / threads_per_block);
     // multiply <<<num_Blocks, threads_per_block>>> (dev_complexResult, dev_phaseFactors, dev_initPhaseFactors,
     //     fftSize, fftCount, resultLen);

    multiply <<<1, 1>>> (dev_complexResult, dev_phaseFactors, dev_initPhaseFactors, fftSize, fftCount, resultLen);

    dev_result = reinterpret_cast<float*>(dev_complexResult);

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    printf("Execution time (without cufftPlan and memory operations): %f ms\n", milliseconds);

    cudaStatus = cudaGetLastError();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "Kernel launch failed\n");
        return cudaStatus;
    }

    cudaStatus = cudaDeviceSynchronize();
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaDeviceSynchronize returned error code %d after launching Kernel!\n", cudaStatus);
        return cudaStatus;
    }

    unsigned endPos = total_signalLen - total_historyLen;
    cudaStatus = cudaMemcpy(dev_history, dev_inComplexSignal + endPos, total_historyLen * sizeof(cufftComplex),
        cudaMemcpyDeviceToDevice);
    if (cudaStatus != cudaSuccess) {
        fprintf(stderr, "cudaMemcpy failed!\n");
        return cudaStatus;
    }

    num_Blocks = ceil((float)fftSize / threads_per_block);
    updateInitPhaseFactors<<<num_Blocks,threads_per_block>>>(dev_initPhaseFactors, signalLen, fftSize);
    
    cudaFree(dev_tensor);

    return cudaStatus;
}
