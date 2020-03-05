import numpy as np
from scipy import signal as sp
import matplotlib.pyplot as plt



def squeeze(arr, size):
    result = np.zeros(size, dtype="complex128")
    k = len(arr) // size
    sum = 0
    for i in range(size):
        for j in range(k):
            sum += arr[i + j*size]
        result[i] = sum
        sum = 0
    return result

def filterbank(inSignal, h, K, F):
    T = len(h)
    N = len(inSignal)
    xh = np.zeros(N, dtype='complex128')
    count = len(range(1, N - T, K))
    ffts = np.zeros((count, F), dtype="complex128")

    for i in range(count):
        start = i*K
        for n in range(T):
            xh[start + n] = inSignal[start + n] * h[n]
        subArr = xh[start: start + T]
        fftArr = squeeze(subArr, F)
        fftResult = (np.fft.fft(fftArr))
        ffts[i, :] = fftResult

    return ffts



def plotSpectrum(signal, title = None):
    signalLen = len(signal)
    magfft_of_signal = np.abs(np.fft.fftshift(np.fft.fft(signal, norm="ortho")))
    freqs = np.arange(-(signalLen // 2), signalLen // 2)
    plt.figure()
    plt.title(title)
    plt.plot(freqs, magfft_of_signal)

def plotSubbands(fft_matrix, step = 1):
    for n in range(0, fft_matrix.shape[1], step):
        plt.figure()
        plt.stem(np.abs(np.fft.ifft(fft_matrix[:,n])), use_line_collection="true")
        plt.ylim((0, 1)) 
        #plt.stem(np.real(fft_matrix[:,n]), use_line_collection="true")
        #plt.ylim((-1, 1)) 
        plt.title("subband #" + str(n))
        
        


signalLen = 1024*8
filterLen = 512
fft_size = filterLen // 64
f_cutoff = 1/(fft_size)
n = np.arange(0, signalLen)

signal = np.exp(1j*2*np.pi*0*n) + 0.5*np.exp(1j*2*np.pi*0.3*n) + 0.3*np.exp(1j*2*np.pi*0.7*n)
#signal = sp.chirp(n, 0, signalLen, 1/4)

plotSpectrum(signal, "fft of the signal")

# plt.figure()
# plt.title("signal")
# plt.stem(signal, use_line_collection=True)

taps = sp.firwin(filterLen, f_cutoff, window="blackman")
#plotSpectrum(taps, "filter AR")

fft_matrix = filterbank(signal, taps, 128, fft_size)

plotSubbands(fft_matrix, 1)

plt.show()