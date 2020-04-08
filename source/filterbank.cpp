#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include <tuple>
#include "filterbank.hpp"
#include "fb_multi_channel_Impl.cuh"

class Filterbank::Filterbank_impl{
public:
    Filterbank_impl(unsigned signalLen, unsigned channelCount, unsigned fftSize, unsigned step,
        unsigned filterLen, float* dev_filterTaps, unsigned threadsPerBlock) : 
    signalLen(signalLen), channelCount(channelCount), fftSize(fftSize), step(step),
    filterLen(filterLen), threadsPerBlock(threadsPerBlock), dev_filterTaps(dev_filterTaps)
    {
        cudaError_t cudaStatus;
        cudaStatus = cudaSetDevice(0);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaSetDevice failed! Do you have a CUDA-capable GPU installed?\n");
        }

        unsigned fftCount = signalLen / step;
        unsigned total_fftSize = fftCount * fftSize;
        resultLen = total_fftSize * channelCount;

        int * nx = new int(fftSize);
        int idist = channelCount * fftSize;
        int odist = fftSize;
        int istride = channelCount, ostride = 1;
        int *inembed = new int(resultLen);
        int *onembed = new int(total_fftSize);
        cufftResult cufftStatus;
        cufftStatus = cufftPlanMany(&plan, 1, nx, inembed, istride, idist,
            onembed, ostride, odist, CUFFT_C2C, fftCount);

        if (cufftStatus != CUFFT_SUCCESS) {
            fprintf(stderr, "cufftPlanMany failed. Error code %d!\n", cufftStatus);
        }

        cufftComplex* phaseFactors = new cufftComplex[total_fftSize];
        getPhaseFactors(phaseFactors, fftSize, fftCount, step, signalLen);

        cudaStatus = cudaMalloc((void**)&dev_phaseFactors, total_fftSize * sizeof(cufftComplex));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
        }

        cudaStatus = cudaMemcpy(dev_phaseFactors, phaseFactors, total_fftSize * sizeof(cufftComplex),
            cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
        }

        cudaStatus = cudaMalloc((void**)&dev_history, (filterLen - 1) * channelCount * sizeof(cufftComplex));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
        }

        cudaStatus = cudaMemset(dev_history, 0, (filterLen - 1) * channelCount);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemset failed!\n");
        }

        packetIndex = 0;
    }

    ~Filterbank_impl()
    {
        cudaFree(dev_phaseFactors);
        cudaFree(dev_filterTaps);
        cudaFree(dev_history);
        cufftDestroy(plan);
    }

    int execute(float * dev_inSignal, float * dev_result){
        int status;
        status = executeImpl(dev_inSignal, signalLen, dev_filterTaps, filterLen, fftSize, step, channelCount,
            dev_result, resultLen, threadsPerBlock, packetIndex, plan, dev_phaseFactors, dev_history);

        packetIndex++;
        return status;
    }

    int getPhaseFactors(cufftComplex * result, unsigned fftSize, unsigned fftCount, unsigned step, unsigned signalLen){
        for (unsigned k = 0; k < fftCount; ++k)
        {
            for (unsigned f = 0; f < fftSize; ++f)
            {
                float arg = -2 * M_PI * f * k * fftCount / signalLen;
                result[k*fftSize + f].x = cosf(arg);
                result[k*fftSize + f].y = sinf(arg);
            }
        }
        return 0;
    }

private:
    unsigned signalLen;
    unsigned long resultLen;
    unsigned channelCount;
    unsigned fftSize;
    unsigned step;
    unsigned filterLen;
    unsigned threadsPerBlock;
    unsigned packetIndex;
    float* filterTaps;
    float* dev_filterTaps;
    cufftComplex* dev_phaseFactors;
    cufftComplex* dev_history;
    cufftHandle plan;
};

Filterbank::Filterbank(unsigned signalLen, unsigned channelCount, unsigned fftSize,
    unsigned step, unsigned filterLen, float* filterTaps, unsigned threadsPerBlock) :
    impl(new Filterbank_impl(signalLen, channelCount, fftSize, step, filterLen, filterTaps, threadsPerBlock)),
    signalLen(signalLen), channelCount(channelCount), fftSize(fftSize),
    step(step), filterLen(filterLen), threadsPerBlock(threadsPerBlock)
{

}

Filterbank::~Filterbank()
{
    delete impl;
    impl = 0;
}

int Filterbank::getOutDim()
{
  return -1; //TO DO
}

int Filterbank::execute(float* dev_inSignal, float* dev_result)
{
    return impl->execute(dev_inSignal, dev_result);
}

