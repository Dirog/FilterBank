struct Dim {
    const unsigned arrSize = 4;
    unsigned* dimension = new unsigned[arrSize];
    Dim(unsigned x, unsigned y, unsigned z, unsigned rank);
};

class Filterbank
{
public:
    Filterbank(unsigned signalLen, unsigned channelCount, unsigned fftSize,
        unsigned step, unsigned filterLen, float* filterTaps, unsigned threadsPerBlock);
    
    ~Filterbank();

    Dim* getOutDim();
    int execute(float * inSignal, float * outSignal);
    
private:
    unsigned signalLen;
    unsigned channelCount;
    unsigned fftSize;
    unsigned step;
    unsigned filterLen;
    unsigned threadsPerBlock;
    Dim* dim;

    class Filterbank_impl;
    Filterbank_impl * impl;
};