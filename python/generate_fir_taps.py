import numpy as np
from scipy import signal as sp

signal_file = open("./files/taps", "w")
fft_size = 256
filterLen = 256

f_cutoff = 1/(fft_size)
taps = sp.firwin(filterLen, f_cutoff)

for tap in taps:
    signal_file.write("%f " % 1)
