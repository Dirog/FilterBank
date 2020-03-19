import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as sp

def _chirp_phase(t, f0, t1, f1, method='linear', vertex_zero=True):
    t = np.asarray(t)
    f0 = float(f0)
    t1 = float(t1)
    f1 = float(f1)
    if method in ['linear', 'lin', 'li']:
        beta = (f1 - f0) / t1
        phase = 2 * np.pi * (f0 * t + 0.5 * beta * t * t)

    elif method in ['quadratic', 'quad', 'q']:
        beta = (f1 - f0) / (t1 ** 2)
        if vertex_zero:
            phase = 2 * np.pi * (f0 * t + beta * t ** 3 / 3)
        else:
            phase = 2 * np.pi * (f1 * t + beta * ((t1 - t) ** 3 - t1 ** 3) / 3)

    elif method in ['logarithmic', 'log', 'lo']:
        if f0 * f1 <= 0.0:
            raise ValueError("For a logarithmic chirp, f0 and f1 must be "
                             "nonzero and have the same sign.")
        if f0 == f1:
            phase = 2 * pi * f0 * t
        else:
            beta = t1 / np.log(f1 / f0)
            phase = 2 * np.pi * beta * f0 * (np.power(f1 / f0, t / t1) - 1.0)

    elif method in ['hyperbolic', 'hyp']:
        if f0 == 0 or f1 == 0:
            raise ValueError("For a hyperbolic chirp, f0 and f1 must be "
                             "nonzero.")
        if f0 == f1:
            phase = 2 * np.pi * f0 * t
        else:
            sing = -f1 * t1 / (f0 - f1)
            phase = 2 * np.pi * (-sing * f0) * np.log(np.abs(1 - t/sing))

    else:
        raise ValueError("method must be 'linear', 'quadratic', 'logarithmic',"
                " or 'hyperbolic', but a value of %r was given." % method)

    return phase

def complex_chirp(t, f0, t1, f1, method='linear', phi=0, vertex_zero=True):
    phase = _chirp_phase(t, f0, t1, f1, method, vertex_zero)
    phi *= np.pi / 180
    return np.exp(1j * (phase + phi))



filter_file = open("./python/files/taps", "w")
signal_file = open("./python/files/signal", "w")


signalLen = 2048
filterLen = 128
fft_size = filterLen // 32
f_cutoff = 1/(fft_size)
n = np.linspace(0, 1, signalLen)
taps = sp.firwin(filterLen, f_cutoff)
taps = taps[::-1]


#signal1 = np.zeros(signalLen, dtype="complex128")
signal2 = np.zeros(signalLen, dtype="complex128")
signal3 = np.zeros(signalLen, dtype="complex128")


signal1 = complex_chirp(n, -32, 1, 32) + 2*complex_chirp(n, -500, 1, -400) + 2*complex_chirp(n, 400, 1, 500) + 3*complex_chirp(n, -900, 1, -800) + 3*complex_chirp(n, 800, 1, 900)
signal2 = 2*complex_chirp(n, 400, 1, 500)
signal3 = 3*complex_chirp(n, -900, 1, -800)

for tap in taps:
    filter_file.write("%f " % tap)

for i in range(signalLen):
    signal_file.write("%f " % signal1[i].real)
    signal_file.write("%f " % signal1[i].imag)

    signal_file.write("%f " % signal2[i].real)
    signal_file.write("%f " % signal2[i].imag)

    signal_file.write("%f " % signal3[i].real)
    signal_file.write("%f " % signal3[i].imag)
    
plt.plot(np.abs(np.fft.fftshift(np.fft.fft(signal3)  / signalLen )))
plt.show()
