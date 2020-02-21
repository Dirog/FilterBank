import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

def filterbank(inSignal, window, K, F):
    T = len(window)
    N = len(inSignal)
    xh = np.zeros(N)
    preFFT = []
    for wndIndx in range(N // T):
        for i in range(T):
            sampleIndx = i + wndIndx*T
            xh[sampleIndx] = inSignal[sampleIndx]*window[i]

            

    # for q in range(N - F):     
    #     sum = 0.0          
    #     for f in range(K):
    #         indx = f + q*K
    #         sum += xh[indx]
    #     preFFT.append(sum)

    # for i in range( N //  ):
    #     preFFT.append(sum(xh[i: i + K))


    ffts = []
    for n in range(N - F):
        ffts.append(sp.fft(preFFT[n: n + F]))

    return ffts, preFFT, xh

#print( filterbank([1,1,1,1,1,1,1,1], [1,2,3,4], 1, 2) )

Fs = 1
dt = 1 / Fs
f1 = 0.01
f2 = 0.1
f3 = 0.4
pi = np.pi
windowSize = 256
t = np.arange(0, 1024, dt)
signal = np.sin(2*pi*f1*t) + np.sin(2*pi*f2*t) + np.sin(2*pi*f3*t)
window = np.ones((windowSize,))

ffts, preFFT, xh = filterbank(signal, window, 256, 256)

plt.figure()
plt.plot(t, signal)


plt.figure()
plt.plot( t, np.abs( sp.fft(signal) ) )


plt.figure()
plt.plot( np.arange(0, len(ffts[0])), np.abs( ffts[0] ) )
plt.show()



