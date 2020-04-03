#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include <math.h>
#include "filterbank.hpp"
#include "fb_multi_channel_Impl.cuh"

class filterbank::filterbank_impl{
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
    cufftHandle plan;

public:
    filterbank_impl(unsigned signalLen, unsigned channelCount, unsigned fftSize, unsigned step, unsigned filterLen, float* filterTaps, unsigned threadsPerBlock)
    {
        this->signalLen = signalLen;
        this->channelCount = channelCount;
        this->fftSize = fftSize;
        this->step = step;
        this->filterLen = filterLen;
        this->filterTaps = filterTaps;
        unsigned fftCount = signalLen / step;
        this->resultLen = 2 * fftSize * fftCount * channelCount;
        this->threadsPerBlock = threadsPerBlock;

        cufftHandle plan;
        int * nx = new int(fftSize);
        int idist = channelCount * fftSize;
        int odist = fftSize;
        int istride = channelCount, ostride = 1;
        int *inembed = new int(resultLen);
        int *onembed = new int(fftSize * fftCount);
        cufftResult cufftStatus;
        cufftStatus = cufftPlanMany(&plan, 1, nx, inembed, istride, idist, onembed, ostride, odist, CUFFT_C2C, fftCount);
        if (cufftStatus != CUFFT_SUCCESS) {
            fprintf(stderr, "cufftPlanMany failed. Error code %d!\n", cufftStatus);
        }
        this->plan = plan;

        cudaError_t cudaStatus;
        cudaStatus = cudaMalloc((float**)&dev_filterTaps, filterLen * sizeof(float));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
        }

        cudaStatus = cudaMemcpy(dev_filterTaps, filterTaps, filterLen * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
        }

        cufftComplex* phaseFactors = new cufftComplex[fftCount * fftSize];

        getPhaseFactors(phaseFactors, fftSize, fftCount, step, signalLen);

        cudaStatus = cudaMalloc((void**)&dev_phaseFactors, fftCount * fftSize * sizeof(cufftComplex));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
        }

        cudaStatus = cudaMemcpy(dev_phaseFactors, phaseFactors, fftCount * fftSize * sizeof(cufftComplex), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
        }
    }

    ~filterbank_impl()
    {
        cudaFree(dev_filterTaps);
        cufftDestroy(plan);
    }

    int execute(float * inSignal, float * result){
        return executeImpl(inSignal, signalLen, dev_filterTaps, filterLen, fftSize, step, channelCount, result, resultLen, threadsPerBlock, plan, dev_phaseFactors);
    }

    int getPhaseFactors(cufftComplex * result, unsigned fftSize, unsigned fftCount, unsigned step, unsigned signalLen){
        const float PI = 3.1415927;
        for (unsigned k = 0; k < fftCount; ++k)
        {
            for (unsigned f = 0; f < fftSize; ++f)
            {
                float arg = -2 * PI * f * k * fftCount / signalLen;
                result[k*fftSize + f].x = cosf(arg);
                result[k*fftSize + f].y = sinf(arg);
            }
        }
        return 0;
    }
};




filterbank::filterbank(unsigned signalLen, unsigned channelCount, unsigned fftSize, unsigned step, unsigned filterLen, float* filterTaps, unsigned threadsPerBlock)
                        : impl(new filterbank_impl(signalLen, channelCount, fftSize, step, filterLen, filterTaps, threadsPerBlock))
{
    this->signalLen = signalLen;
    this->channelCount = channelCount;
    this->fftSize = fftSize;
}

filterbank::~filterbank()
{
    delete impl;
    impl = 0;
}

unsigned * filterbank::getOutDim()
{
    return new unsigned[3] {signalLen, channelCount, fftSize}; //?
}

int filterbank::execute(float * inSignal, float * result)
{
    return impl->execute(inSignal, result);
}

