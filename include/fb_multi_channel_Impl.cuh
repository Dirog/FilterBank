int executeImpl(float* inSignal, unsigned signalLen, float* dev_filterTaps, unsigned filterLen,
	unsigned fftSize, unsigned step, unsigned channelCount, float* result, unsigned long resultLen,
	unsigned threads_per_block, cufftHandle plan, cufftComplex* dev_phaseFactors, cufftComplex* dev_history);
