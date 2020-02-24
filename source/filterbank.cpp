//TO DO: Pimpl


class filterbank
{
private:
    unsigned inSignalLen;
    unsigned inChannelCount;
    unsigned outSubBandsCount;
    unsigned windowStep;
    unsigned filterTapsCount;
    float* filterTaps;
public:
    filterbank(unsigned inSignalLen, unsigned inChannelCount,
                        unsigned outSubBandsCount, unsigned windowStep,
                        unsigned filterTapsCount, float* filterTaps);
    ~filterbank();
    unsigned int* getOutDim();
    int execute(float * inSignal, float * outSignal);
};

filterbank::filterbank(unsigned inSignalLen, unsigned inChannelCount,
                        unsigned outSubBandsCount, unsigned windowStep,
                        unsigned filterTapsCount, float* filterTaps)
{
    this->inSignalLen = inSignalLen;
    this->inChannelCount = inChannelCount;
    this->outSubBandsCount = outSubBandsCount;
    this->windowStep = windowStep;
    this->filterTapsCount = filterTapsCount;
    this->filterTaps = filterTaps;
}

filterbank::~filterbank()
{
}

unsigned * filterbank::getOutDim(){
    return new unsigned[3] {inSignalLen, inChannelCount, outSubBandsCount};
}

int filterbank::execute(float * inSignal, float * outSignal){
    return -1;
}