class filterbank
{
private:
    unsigned signalLen;
    unsigned long resultLen;
    unsigned channelCount;
    unsigned fftSize;
    unsigned step;
    unsigned filterLen;
    unsigned threadsPerBlock;
    float* filterTaps;

    class filterbank_impl;
    filterbank_impl * impl;

public:
    filterbank(unsigned signalLen, unsigned channelCount, unsigned fftSize, unsigned step,
               unsigned filterLen, float* filterTaps, unsigned threadsPerBlock);
    ~filterbank();

    unsigned * getOutDim();
    int execute(float * inSignal, float * outSignal);
};