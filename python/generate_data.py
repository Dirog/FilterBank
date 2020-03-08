import numpy as np
from scipy import signal as sp

filter_file = open("./files/taps", "w")
signal_file = open("./files/signal", "w")


signalLen = 1024*8
filterLen = 512
fft_size = filterLen // 64
f_cutoff = 1/(fft_size)
n = np.arange(0, signalLen)

taps = sp.firwin(filterLen, f_cutoff, window="blackman")
signal = np.exp(1j*2*np.pi*0*n) + 0.5*np.exp(1j*2*np.pi*0.3*n) + 0.3*np.exp(1j*2*np.pi*0.7*n)

for tap in taps:
    filter_file.write("%f " % tap)

for sample in signal:
    signal_file.write("%f " % sample.real)
    signal_file.write("%f " % sample.imag)
    
