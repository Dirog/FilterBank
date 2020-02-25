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
    ffts_list = []

    for i in range(N//T):
        start = i*T
        for n in range(T):
            xh[start + n] = inSignal[start + n] * h[n]

    for i in range(count):
        start = i*K
        subArr = xh[start: start + T]
        fftArr = squeeze(subArr, F)
        fftResult = np.fft.fftshift(np.fft.fft(fftArr))
        ffts[i, :] = fftResult
        ffts_list.append(fftResult)

    return ffts

def plotSpectrum(signal, title = None):
    signalLen = len(signal)
    magfft_of_signal = np.abs(np.fft.fftshift(np.fft.fft(signal, norm="ortho")))
    freqs = np.arange(-(signalLen // 2), signalLen // 2)
    plt.figure()
    plt.title(title)
    plt.plot(freqs, magfft_of_signal)



f = 0.1
signalLen = 1024
filterLen = 256
fft_size = 256
f_cutoff = 1/(fft_size)
n = np.arange(0, signalLen)
#signal = np.sin(2*np.pi*f*n)
#signal = np.exp(1j*2*np.pi*f*n)
signal = sp.chirp(n, 0, 1024, 0.5)
plotSpectrum(signal, "fft of the signal")

plt.figure()
plt.title("signal")
plt.plot(signal)

taps = sp.firwin(filterLen, f_cutoff)
#plotSpectrum(taps, "filter AR")

fft_matrix = filterbank(signal, taps, 8, fft_size)
plt.matshow(np.abs(fft_matrix))
plt.title("spectrogram")

plt.show()