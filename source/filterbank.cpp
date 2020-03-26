#include "filterbank.hpp"
#include "fb_multi_channel_Impl.cuh"

filterbank::filterbank(unsigned signalLen, unsigned channelCount,
                        unsigned fftSize, unsigned step,
                        unsigned filterLen, float* filterTaps, unsigned threads_per_block)
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
}

filterbank::~filterbank()
{
}

unsigned * filterbank::getOutDim(){
    return new unsigned[3] {signalLen, channelCount, fftSize};
}

int filterbank::execute(float * inSignal, float * result)
{
    executeImpl(inSignal, signalLen, filterTaps, filterLen,
                fftSize, step, channelCount, result, resultLen, threads_per_block);

    return 0;
}