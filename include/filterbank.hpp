class filterbank
{
private:
    unsigned signalLen;
    unsigned long resultLen;
    unsigned channelCount;
    unsigned fftSize;
    unsigned step;
    unsigned filterLen;
    unsigned threads_per_block;
    float* filterTaps;

    class fb_impl;
    fb_impl * impl;
public:
    filterbank(unsigned signalLen, unsigned channelCount, unsigned fftSize, unsigned step,
                        unsigned filterLen, float* filterTaps, unsigned threads_per_block);
    ~filterbank();
    unsigned * getOutDim();
    int execute(float * inSignal, float * outSignal);
};