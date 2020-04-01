#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
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
    unsigned threads_per_block;
    float* filterTaps;
    float* dev_filterTaps;
    cufftHandle plan;

public:
    filterbank_impl(unsigned signalLen, unsigned channelCount, unsigned fftSize, unsigned step, unsigned filterLen, float* filterTaps, unsigned threads_per_block)
    {
        this->signalLen = signalLen;
        this->channelCount = channelCount;
        this->fftSize = fftSize;
        this->step = step;
        this->filterLen = filterLen;
        this->filterTaps = filterTaps;
        unsigned fftCount = ((signalLen - filterLen) / (step)) + 1;
        this->resultLen = 2 * fftSize * fftCount * channelCount;
        this->threads_per_block = threads_per_block;

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
            fprintf(stderr, "cudaMalloc failed! 3\n");
        }

        cudaStatus = cudaMemcpy(dev_filterTaps, filterTaps, filterLen * sizeof(float), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed! 6\n");
        }

        this->dev_filterTaps = dev_filterTaps;
    }

    ~filterbank_impl()
    {
        cudaFree(dev_filterTaps);
        cufftDestroy(plan);
    }

    int execute(float * inSignal, float * result){
        return executeImpl(inSignal, signalLen, dev_filterTaps, filterLen, fftSize, step, channelCount, result, resultLen, threads_per_block, plan);
    }
};




filterbank::filterbank(unsigned signalLen, unsigned channelCount, unsigned fftSize, unsigned step, unsigned filterLen, float* filterTaps, unsigned threads_per_block)
                        : impl(new filterbank_impl(signalLen, channelCount, fftSize, step, filterLen, filterTaps, threads_per_block))
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
    return new unsigned[3] {signalLen, channelCount, fftSize};
}

int filterbank::execute(float * inSignal, float * result)
{
    return impl->execute(inSignal, result);
}

