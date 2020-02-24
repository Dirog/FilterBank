import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


def squeeze(arr, size):
    return arr

def filterbank(inSignal, h, K, F):
    T = len(h)
    N = len(inSignal)
    xh = np.zeros(N, dtype='complex128')
    subbandsCount = len(range(1, N - T, K))
    ffts = []

    for i in range(N//T):
        start = i*T
        for n in range(T):
            xh[start + n] = inSignal[start + n] * h[n]

    for i in range(subbandsCount):
        start = i*K
        subArr = xh[start: start + T]
        fftArr = squeeze(subArr, F)
        fftResult = np.fft.fft(fftArr)
        ffts.append(fftResult)

    return ffts, xh

#print( filterbank([1,2,3,4,5,6,7,8], [1,2,3,4], 2, 2) )

Fs = 1
dt = 1 / Fs
f1 = 0.1
f2 = 0.3
pi = np.pi

t = np.arange(0, 256, dt)
signal = np.exp(1j*2*pi*f1*t) + np.exp(1j*2*pi*f2*t)
#signal = np.ones(256)
#window = np.ones((windowSize,))
#num of taps 64:
h = [-0.000757704804076121, -0.000683755534102993, 1.04013175375723e-18, 0.000887311875128924, 0.00124496760746086, 0.000489892468210470, -0.00113486895422194, -0.00235077136409080, -0.00167592915215251, 0.00106260883184079, 0.00391854193278563, 0.00396740281901117, -4.26133990229082e-18, -0.00551550434497760, -0.00758139982354959, -0.00286691775842712, 0.00632187885183466, 0.0124303555981707, 0.00842571381000707, -0.00510211539104987, -0.0180847562922352, -0.0177387975421038, 9.31450996394886e-18, 0.0238286081418624, 0.0327809284273518, 0.0125992516068269, -0.0287989217414898, -0.0602739485764063, -0.0451890047405531, 0.0321755674766605, 0.150201524963259, 0.257245810605184, 0.300348062007683, 0.257245810605184, 0.150201524963259, 0.0321755674766605, -0.0451890047405531, -0.0602739485764063, -0.0287989217414898, 0.0125992516068269, 0.0327809284273518, 0.0238286081418624, 9.31450996394886e-18, -0.0177387975421038, -0.0180847562922352, -0.00510211539104987, 0.00842571381000707, 0.0124303555981707, 0.00632187885183466, -0.00286691775842712, -0.00758139982354959, -0.00551550434497760, -4.26133990229082e-18, 0.00396740281901117, 0.00391854193278563, 0.00106260883184079, -0.00167592915215251, -0.00235077136409080, -0.00113486895422194, 0.000489892468210470, 0.00124496760746086, 0.000887311875128924, 1.04013175375723e-18, -0.000683755534102993]
ffts, xh = filterbank(signal, h, 32, 64)

plt.figure()
plt.title("signal")
plt.plot(t, signal)

plt.figure()
plt.title("multiplication result x*h")
plt.plot(t, xh)


plt.figure()
plt.title("fft of the signal")
plt.plot( t, np.abs( sp.fft.fft(signal) ) )


for n in range(len(ffts)):
    plt.figure()
    plt.title("subband #" + str(n + 1))
    plt.stem( np.abs(ffts[n]) )
plt.show()



