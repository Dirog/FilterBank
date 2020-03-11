import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp

filter_file = open("../python/files/taps", "w")
signal_file = open("../python/files/signal", "w")


signalLen = 1024*8
filterLen = 128
fft_size = filterLen // 16
f_cutoff = 1/fft_size
n = np.arange(0, signalLen)

taps = sp.firwin(filterLen, f_cutoff)
signal = np.zeros(signalLen, dtype="complex128")


signal = np.sin(2*np.pi*0.3*n)

for tap in taps:
    filter_file.write("%f " % tap)

for sample in signal:
    signal_file.write("%f " % sample.real)
    signal_file.write("%f " % sample.imag)
    
# plt.stem(np.abs(np.fft.fft(signal)))
# plt.show()