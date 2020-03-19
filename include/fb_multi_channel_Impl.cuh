#include <cuda_runtime.h>
#include <cufft.h>


int executeImpl(float* inSignal, unsigned signalLen, float* filterTaps, unsigned filterLen,
                    unsigned fftSize, unsigned step, unsigned channelCount, float* result, unsigned long resultLen);
                    
__global__ void mupltiply_sum(cufftComplex* signal, cufftComplex* resultVec, float* filterTaps, int k, int step,
									int filterLen, int channelCount);