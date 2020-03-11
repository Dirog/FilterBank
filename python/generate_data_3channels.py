import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp

filter_file = open("../python/files/taps", "w")
signal_file = open("../python/files/signal", "w")


signalLen = 1024*8
filterLen = 128
fft_size = filterLen // 16
f_cutoff = 1/(fft_size)
n = np.arange(0, signalLen)

taps = sp.firwin(filterLen, f_cutoff)
signal1 = np.zeros(signalLen, dtype="complex128")
signal2 = np.zeros(signalLen, dtype="complex128")
signal3 = np.zeros(signalLen, dtype="complex128")


signal1 = np.sin(2*np.pi*0.1*n)
signal2 = np.sin(2*np.pi*0.2*n)
signal3 = np.sin(2*np.pi*0.4*n)

for tap in taps:
    filter_file.write("%f " % tap)

for i in range(signalLen):
    signal_file.write("%f " % signal1[i].real)
    signal_file.write("%f " % signal1[i].imag)

    signal_file.write("%f " % signal2[i].real)
    signal_file.write("%f " % signal2[i].imag)

    signal_file.write("%f " % signal3[i].real)
    signal_file.write("%f " % signal3[i].imag)
    
# plt.stem(np.abs(np.fft.fft(signal)))
# plt.show()