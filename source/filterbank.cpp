//TO DO: Pimpl
#include "../include/filterbank.hpp"
#include "../include/fb_multi_channel_Impl.cuh"

filterbank::filterbank(unsigned signalLen, unsigned channelCount,
                        unsigned fftSize, unsigned step,
                        unsigned filterLen, float* filterTaps)
{
    this->signalLen = signalLen;
    this->channelCount = channelCount;
    this->fftSize = fftSize;
    this->step = step;
    this->filterLen = filterLen;
    this->filterTaps = filterTaps;
    unsigned fftCount = ((signalLen / 2 - filterLen) / (step)) + 1;
    this->resultLen = 2 * fftSize * fftCount * channelCount;
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
                fftSize, step, channelCount, result, resultLen);

    return 0;
}