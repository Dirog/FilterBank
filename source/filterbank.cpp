#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include "filterbank.hpp"
#include "fb_multi_channel_Impl.cuh"

class Filterbank::Filterbank_impl{
private:
    unsigned signalLen;
    unsigned long resultLen;
    unsigned channelCount;
    unsigned fftSize;
    unsigned step;
    unsigned filterLen;
    unsigned threadsPerBlock;
    float* filterTaps;
    float* dev_filterTaps;
    cufftComplex* dev_phaseFactors;
    cufftComplex* dev_history;
    cufftHandle plan;

public:
    Filterbank_impl(unsigned signalLen, unsigned channelCount, unsigned fftSize, unsigned step,
        unsigned filterLen, float* filterTaps, unsigned threadsPerBlock) :
    signalLen(signalLen), channelCount(channelCount), fftSize(fftSize),
    step(step), filterLen(filterLen), threadsPerBlock(threadsPerBlock)
    {
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

        cudaError_t cudaStatus;
        cudaStatus = cudaMalloc((float**)&dev_filterTaps, filterLen * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
        }

        cudaStatus = cudaMemcpy(dev_filterTaps, filterTaps, filterLen * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
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
    }

    ~Filterbank_impl()
    {
        cudaFree(dev_phaseFactors);
        cudaFree(dev_filterTaps);
        cudaFree(dev_history);
        cufftDestroy(plan);
    }

    int execute(float * inSignal, float * result){
        return executeImpl(inSignal, signalLen, dev_filterTaps, filterLen, fftSize, step, channelCount,
            result, resultLen, threadsPerBlock, plan, dev_phaseFactors, dev_history);
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

unsigned * Filterbank::getOutDim()
{
    return new unsigned[3] {signalLen, channelCount, fftSize}; //?
}

int Filterbank::execute(float * inSignal, float * result)
{
    return impl->execute(inSignal, result);
}

