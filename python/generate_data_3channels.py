import numpy as np
import matplotlib.pyplot as plt
import chirp as ch
from scipy import signal as sp


metadata_file = open("../python/files/metadata", "w")

channelCount = 3
signalLen = 10000000
fft_size = 2401
filterLen = fft_size * 12
step = 10000

print(signalLen // 2)
print("C = " + str(channelCount) + ", N = " + str(signalLen) + ", T = " + str(filterLen) + 
    ", F = " + str(fft_size) + ", K = " + str(step))
metadata_file.write('%d %d %d %d %d' % (channelCount, signalLen // 2, filterLen, fft_size, step))

f_cutoff = 1/(fft_size)

n = np.linspace(0, 1, signalLen)

taps = sp.firwin(filterLen, f_cutoff, window='boxcar')
taps = taps[::-1]
taps = taps.astype("float32")

signal1 = 2*ch.complex_chirp(n, -0.00001*signalLen, 1, 0.00001*signalLen) + 2*ch.complex_chirp(n, (0.71-0.00004)*signalLen, 1, (0.71+0.00001)*signalLen)
signal2 = 1*ch.complex_chirp(n, (1/4-0.00001)*signalLen, 1, (1/4+0.00001)*signalLen)
signal3 = 2*ch.complex_chirp(n, (0.21-0.00001)*signalLen, 1, (0.21+0.00004)*signalLen)

signal1 = signal1.astype("complex64")
signal2 = signal2.astype("complex64")
signal3 = signal3.astype("complex64")

np.asarray(taps).tofile("../python/files/taps")

vector1 = []
for i in range(0, signalLen // 2):
    vector1.append(signal1[i])
    vector1.append(signal2[i])
    vector1.append(signal3[i])
    
np.asarray(vector1).tofile("../python/files/signal1")

vector2 = []
for i in range(signalLen // 2, signalLen):
    vector2.append(signal1[i])
    vector2.append(signal2[i])
    vector2.append(signal3[i])
    
np.asarray(vector2).tofile("../python/files/signal2")



#signals = [signal1, signal2, signal3]
#plt.magnitude_spectrum(taps, window = sp.get_window("boxcar", filterLen), scale = "dB")
# plt.xlim((0, 20*f_cutoff))
# plt.ylim((-200, -80))
# for i in range(len(signals)):
#     fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
#     fig.suptitle("signal #" + str(i+1)) 
#     axes[0].magnitude_spectrum(signals[i], window = sp.get_window("boxcar", signalLen), scale = "dB")
#     axes[1].phase_spectrum(signals[i], window = sp.get_window("boxcar", signalLen))

# plt.show()