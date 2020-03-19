class filterbank
{
private:
    unsigned signalLen;
    unsigned long resultLen;
    unsigned channelCount;
    unsigned fftSize;
    unsigned step;
    unsigned filterLen;
    float* filterTaps;
public:
    filterbank(unsigned signalLen, unsigned channelCount,
                        unsigned fftSize, unsigned step,
                        unsigned filterLen, float* filterTaps);
    ~filterbank();
    unsigned * getOutDim();
    int execute(float * inSignal, float * outSignal);
};