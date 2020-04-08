int executeImpl(float* dev_inSignal, unsigned signalLen, float* dev_filterTaps, unsigned filterLen, unsigned fftSize,
    unsigned step, unsigned channelCount, float* dev_result, unsigned long resultLen,unsigned threads_per_block,
    unsigned packetIndex, cufftHandle plan, cufftComplex* dev_phaseFactors, cufftComplex* dev_history);