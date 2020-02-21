class filterbank
{
private:
    unsigned int inSignalLen;
    unsigned int inChannelCount;
    unsigned int outSubBandsCount;
    unsigned windowStep;
    unsigned int filterTapsCount;
    float* filterTaps;
public:
    filterbank(unsigned int inSignalLen, unsigned int inChannelCount,
                        unsigned int outSubBandsCount, unsigned int windowStep,
                        unsigned int filterTapsCount, float* filterTaps);
    ~filterbank();
    unsigned int* getOutDim();
    float * execute(float * inSignal, float * outSignal);
};