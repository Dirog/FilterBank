import numpy as np
import scipy.signal as sp 
import matplotlib.pyplot as plt

matadata_file = open("../python/files/metadata", "r")

metadata = list(map(int, matadata_file.readline().split()))
channelCount = metadata[0]
signalLen = metadata[1]
filterLen = metadata[2]
fftSize = metadata[3]
step = metadata[4]

count = signalLen // step

print("C = " + str(channelCount) + ", N = " + str(signalLen) + ", T = " + str(filterLen) + 
    ", F = " + str(fftSize) + ", K = " + str(step) + ", fft count = " + str(count))

tensor1 = np.zeros((count, fftSize, channelCount), dtype="complex64")
tensor2 = np.zeros((count, fftSize, channelCount), dtype="complex64")

vector1 = np.fromfile("../python/files/result1", dtype="float32")
vector2 = np.fromfile("../python/files/result2", dtype="float32")

i = 0
for c in range(channelCount):
    for n in range(count):
        for f in range(fftSize):
            tensor1[n, f, c] = complex(float(vector1[2*i]), float(vector1[2*i+1]))
            tensor2[n, f, c] = complex(float(vector2[2*i]), float(vector2[2*i+1]))
            i = i + 1


for channel in range(channelCount):
	for i in range(tensor1.shape[1]):
		tensorSlice1 = tensor1[:,i,channel]
		tensorSlice2 = tensor2[:,i,channel]
		tensorSlice = np.asarray( tensorSlice1.tolist() + tensorSlice2.tolist() )
		signalEnergy = sum(np.abs(i)*np.abs(i) for i in tensorSlice)
		if signalEnergy > 50:
		    fig = plt.figure(constrained_layout=True)
		    fig.suptitle("subband #" + str(i + 1) + ". Channel:" + str(channel + 1)) 
		    gs = fig.add_gridspec(4, 2)

		    f_re_ax = fig.add_subplot(gs[0, :-1])
		    f_re_ax.set_title('Real part of the signal')

		    f_im_ax = fig.add_subplot(gs[1, :-1])
		    f_im_ax.set_title('Imaginary part of the signal')

		    f_inst_freq_ax = fig.add_subplot(gs[2, :-1])
		    f_inst_freq_ax.set_title('Instantaneous frequency')

		    f_inst_phs_ax = fig.add_subplot(gs[3, :-1])
		    f_inst_phs_ax.set_title('Instantaneous phase')

		    f_spec_ax = fig.add_subplot(gs[0:2, 1:])
		    f_spec_ax.set_title('Magnitude spectrum')
		    f_spec_ax.grid()

		    f_phs_ax = fig.add_subplot(gs[2:, 1:])
		    f_phs_ax.set_title('Phase spectrum')

		    f_spec_ax.magnitude_spectrum(tensorSlice, window = sp.get_window("boxcar", tensorSlice.shape[0])) #, scale="dB"
		    f_phs_ax.phase_spectrum(tensorSlice, window = sp.get_window("boxcar", tensorSlice.shape[0]))

		    f_re_ax.plot(np.real(tensorSlice))
		    f_im_ax.plot(np.imag(tensorSlice))
		    f_re_ax.grid(which='both')
		    f_im_ax.grid(which='both')

		    fs = 2 * signalLen
		    analytic_signal = sp.hilbert(np.real(tensorSlice))
		    amplitude_envelope = np.abs(analytic_signal)
		    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
		    instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

		    f_inst_phs_ax.plot(instantaneous_phase)
		    f_inst_freq_ax.plot(instantaneous_frequency)

plt.show()
