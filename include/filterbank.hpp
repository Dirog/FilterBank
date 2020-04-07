#include <tuple>

class Filterbank
{
public:
    Filterbank(unsigned signalLen, unsigned channelCount, unsigned fftSize,
        unsigned step, unsigned filterLen, float* filterTaps, unsigned threadsPerBlock);
    
    ~Filterbank();

    std::tuple<unsigned, unsigned, unsigned> getOutDim();
    int execute(float * inSignal, float * outSignal);
    
private:
    unsigned signalLen;
    unsigned channelCount;
    unsigned fftSize;
    unsigned step;
    unsigned filterLen;
    unsigned threadsPerBlock;

    class Filterbank_impl;
    Filterbank_impl * impl;
};