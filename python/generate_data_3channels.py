import numpy as np
import matplotlib.pyplot as plt
import chirp as ch
from scipy import signal as sp


filter_file = open("../python/files/taps", "w")
signal_file = open("../python/files/signal", "w")
metadata_file = open("../python/files/metadata", "w")

channelCount = 3
signalLen = 1024 * 128
filterLen = 1024
fft_size = 4
step = 4

print("C = " + str(channelCount) + ", N = " + str(signalLen) + ", T = " + str(filterLen) + 
    ", F = " + str(fft_size) + ", K = " + str(step))
metadata_file.write('%d %d %d %d %d' % (channelCount, signalLen, filterLen, fft_size, step))

f_cutoff = 1/(fft_size)

n = np.linspace(0, 1, signalLen)
taps = sp.firwin(filterLen, f_cutoff)
taps = taps[::-1]

signal1 = ch.complex_chirp(n, -100*128, 1, 100*128) + 2*ch.complex_chirp(n, -320*128, 1, -150*128)
signal2 = 2*ch.complex_chirp(n, 400, 1, 500)
signal3 = 3*ch.complex_chirp(n, 700, 1, 800)

signals = [signal1, signal2, signal3]

for tap in taps:
    filter_file.write("%f " % tap)

for i in range(signalLen):
    signal_file.write("%f " % signal1[i].real)
    signal_file.write("%f " % signal1[i].imag)

    signal_file.write("%f " % signal2[i].real)
    signal_file.write("%f " % signal2[i].imag)

    signal_file.write("%f " % signal3[i].real)
    signal_file.write("%f " % signal3[i].imag)
    


for i in range(len(signals)):
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
    fig.suptitle("signal #" + str(i+1)) 
    axes[0].magnitude_spectrum(signals[i], window = sp.get_window("boxcar", signalLen))
    axes[1].phase_spectrum(signals[i], window = sp.get_window("boxcar", signalLen))

plt.show()
