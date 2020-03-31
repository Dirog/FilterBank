import numpy as np
import matplotlib.pyplot as plt
import chirp as ch
from scipy import signal as sp


metadata_file = open("../python/files/metadata", "w")

channelCount = 3
signalLen = 20000000
fft_size = 1024
filterLen = 1024 * 16
step = signalLen // 10000

print("C = " + str(channelCount) + ", N = " + str(signalLen) + ", T = " + str(filterLen) + 
    ", F = " + str(fft_size) + ", K = " + str(step))
metadata_file.write('%d %d %d %d %d' % (channelCount, signalLen, filterLen, fft_size, step))

f_cutoff = 1/(fft_size)

n = np.linspace(0, 1, signalLen)

taps = sp.firwin(filterLen, f_cutoff)
taps = taps[::-1]
taps = taps.astype("float32")

signal1 = 2*ch.complex_chirp(n, -0.00001*signalLen, 1, 0.00001*signalLen) + ch.complex_chirp(n, (0.5-0.00001)*signalLen, 1, (0.5+0.00001)*signalLen)
signal2 = 2*ch.complex_chirp(n, (1/4-0.00001)*signalLen, 1, (1/4+0.00001)*signalLen)
signal3 = 2*ch.complex_chirp(n, (0.21-0.00001)*signalLen, 1, (0.21+0.00001)*signalLen)

signal1 = signal1.astype("complex64")
signal2 = signal2.astype("complex64")
signal3 = signal3.astype("complex64")

signals = [signal1, signal2, signal3]

np.asarray(taps).tofile("../python/files/taps")

vector = []
for i in range(signalLen):
    vector.append(signal1[i])
    vector.append(signal2[i])
    vector.append(signal3[i])
    
np.asarray(vector).tofile("../python/files/signal")

#plt.magnitude_spectrum(taps, window = sp.get_window("boxcar", filterLen))
# for i in range(len(signals)):
#     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
#     fig.suptitle("signal #" + str(i+1)) 
#     axes[0].magnitude_spectrum(signals[i], window = sp.get_window("boxcar", signalLen))
#     axes[1].phase_spectrum(signals[i], window = sp.get_window("boxcar", signalLen))

plt.show()