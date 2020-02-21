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
    float * execute(float * inSignal, float * outSignal);
};

filterbank::filterbank(unsigned inSignalLen, unsigned inChannelCount,
                        unsigned outSubBandsCount, unsigned windowStep,
                        unsigned filterTapsCount, float* filterTaps)
{
    filterbank::inSignalLen = inSignalLen;
    filterbank::inChannelCount = inChannelCount;
    filterbank::outSubBandsCount = outSubBandsCount;
    filterbank::windowStep = windowStep;
    filterbank::filterTapsCount = filterTapsCount;
    filterbank::filterTaps = filterTaps;
}

filterbank::~filterbank()
{
}

unsigned * filterbank::getOutDim(){
    return new unsigned[3] {inSignalLen, inChannelCount, outSubBandsCount};
}

float * filterbank::execute(float * inSignal, float * outSignal){ //float32
    
}
