#include <stdio.h>
#include <cuda_runtime.h>
#include <cufft.h>
#include "filterbank.hpp"
#include "fb_multi_channel_Impl.cuh"

int highestPowerof2(int n) 
{ 
    int res = 0; 
    for (int i = n; i >= 1; i--) 
    { 
        if ((i & (i-1)) == 0) 
        { 
            res = i; 
            break; 
        } 
    } 
    return res; 
}

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

        int* nx = new int(fftSize);
        int idist = channelCount * fftSize;
        int odist = fftSize;
        int istride = channelCount, ostride = 1;
        int* inembed = new int(resultLen);
        int* onembed = new int(total_fftSize);
        cufftResult cufftStatus;
        cufftStatus = cufftPlanMany(&plan, 1, nx, inembed, istride, idist,
            onembed, ostride, odist, CUFFT_C2C, fftCount);

        if (cufftStatus != CUFFT_SUCCESS) {
            fprintf(stderr, "cufftPlanMany failed. Error code %d!\n", cufftStatus);
        }

        cufftComplex* phaseFactors = new cufftComplex[total_fftSize];
        cufftComplex* initPhaseFactors = new cufftComplex[fftSize];
        getPhaseFactors(phaseFactors, fftSize, fftCount, step, signalLen, filterLen);
        getInitPhaseFactors(initPhaseFactors, fftSize, fftCount);

        cudaStatus = cudaMalloc((void**)&dev_phaseFactors, total_fftSize * sizeof(cufftComplex));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
        }

        cudaStatus = cudaMalloc((void**)&dev_initPhaseFactors, fftSize * sizeof(cufftComplex));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
        }

        cudaStatus = cudaMemcpy(dev_phaseFactors, phaseFactors, total_fftSize * sizeof(cufftComplex),
            cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
        }

        cudaStatus = cudaMemcpy(dev_initPhaseFactors, initPhaseFactors, fftSize * sizeof(cufftComplex),
            cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
        }

        cudaStatus = cudaMalloc((void**)&dev_history, (filterLen - 1) * channelCount * sizeof(cufftComplex));
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMalloc failed!\n");
        }

        cufftComplex* zeros = new cufftComplex[filterLen - 1]();
        cudaStatus = cudaMemcpy(dev_history, zeros, (filterLen - 1) * sizeof(cufftComplex), cudaMemcpyHostToDevice);
        if (cudaStatus != cudaSuccess) {
            fprintf(stderr, "cudaMemcpy failed!\n");
        }

        if (threadsPerBlock > fftSize)
        {
            threadsPerBlock = highestPowerof2(fftSize);
        }
    }

    ~Filterbank_impl()
    {
        cudaFree(dev_phaseFactors);
        cudaFree(dev_filterTaps);
        cudaFree(dev_history);
        cufftDestroy(plan);
    }

    int execute(float* dev_inSignal, float* dev_result){
        int status;
        status = executeImpl(dev_inSignal, signalLen, dev_filterTaps, filterLen, fftSize, step, channelCount,
            dev_result, resultLen, threadsPerBlock, plan, dev_phaseFactors, dev_history, dev_initPhaseFactors);

        return status;
    }

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
    cufftComplex* dev_initPhaseFactors;
    cufftComplex* dev_history;
    cufftHandle plan;

    int getPhaseFactors(cufftComplex* result, unsigned fftSize, unsigned fftCount, unsigned step, unsigned signalLen, unsigned filterLen)
    {
        for (unsigned k = 0; k < fftCount; ++k)
        {
            for (unsigned f = 0; f < fftSize; ++f)
            {
                double arg = -2 * M_PI * f * k * step / fftSize;
                result[k*fftSize + f].x = cos(arg);
                result[k*fftSize + f].y = sin(arg);
            }
        }
        return 0;
    }

    int getInitPhaseFactors(cufftComplex* initPhaseFactors, unsigned fftSize, unsigned fftCount)
    {
        for (unsigned f = 0; f < fftSize; ++f)
        {
            initPhaseFactors[f].x = 1;
            initPhaseFactors[f].y = 0;
        }
        return 0;
    }
};

Dim::Dim(unsigned x, unsigned y, unsigned z, unsigned rank){
    dimension = new unsigned[arrSize] {x, y, z, rank};
}

Filterbank::Filterbank(unsigned signalLen, unsigned channelCount, unsigned fftSize,
    unsigned step, unsigned filterLen, float* filterTaps, unsigned threadsPerBlock) :
    impl(new Filterbank_impl(signalLen, channelCount, fftSize, step, filterLen, filterTaps, threadsPerBlock)),
    signalLen(signalLen), channelCount(channelCount), fftSize(fftSize),
    step(step), filterLen(filterLen), threadsPerBlock(threadsPerBlock)
{
    int rank = 2;
    dim = new Dim(signalLen / step, channelCount, fftSize, rank);
}

Filterbank::~Filterbank()
{
    delete impl;
    impl = 0;
}

Dim* Filterbank::getOutDim()
{
  return dim;
}

int Filterbank::execute(float* dev_inSignal, float* dev_result)
{
    return impl->execute(dev_inSignal, dev_result);
}

