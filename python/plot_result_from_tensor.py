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

tensor = np.zeros((count, fftSize, channelCount), dtype="complex64")

vector = np.fromfile("../python/files/result", dtype="float32")
#print(vector)

i = 0
for c in range(channelCount):
    for n in range(count):
        for f in range(fftSize):
            tensor[n, f, c] = complex(float(vector[2*i]), float(vector[2*i+1]))
            i = i + 1


def plotSubbands(tensor, channel):
    for i in range(tensor.shape[1]): #tensor.shape[1]
        tensorSlice = tensor[:,i,channel]
        if np.max(np.abs(tensorSlice)) > 0:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
            fig.suptitle("subband #" + str(i) + ". Channel:" + str(channel + 1)) 
            axes[0].magnitude_spectrum(tensorSlice, window = sp.get_window("boxcar", count), scale="dB")
            axes[1].phase_spectrum(tensorSlice, window = sp.get_window("boxcar", count))
            #axes[0].set_ylim((0, 0.1))
            #fig.savefig('channel_%d_subband_%d.png' % ((channel + 1), i))

def plotSubband(tensor, channel, subband):
    tensorSlice = tensor[:,subband,channel]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
    fig.suptitle("subband #" + str(subband) + ". Channel:" + str(channel + 1)) 
    axes[0].magnitude_spectrum(tensorSlice, window = sp.get_window("boxcar", count))
    axes[1].phase_spectrum(tensorSlice, window = sp.get_window("boxcar", count))
    #axes[0].set_ylim((0, 0.4))
    #fig.savefig('channel_%d_subband_%d.png' % ((channel + 1), i))

def plotSignal(tensor, channel):
    for i in range(tensor.shape[1]): #
        tensorSlice = tensor[:,i,channel]
        signalEnergy = sum(np.abs(i)*np.abs(i) for i in tensorSlice)
        if signalEnergy > 10:
            fig, axes = plt.subplots(nrows=4, ncols=1, figsize=(7, 7))
            axes[0].plot(np.real(tensorSlice))
            axes[0].set_title("Re")
            axes[1].plot(np.imag(tensorSlice))
            axes[1].set_title("Im")

            analytic_signal = sp.hilbert(np.real(tensorSlice))
            amplitude_envelope = np.abs(analytic_signal)
            instantaneous_phase = np.unwrap(np.angle(analytic_signal))

            fs = signalLen
            instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)
            axes[2].set_title("Instantaneous phase")
            axes[2].plot(instantaneous_phase)
            axes[3].set_title("Instantaneous frequency")
            axes[3].plot(instantaneous_frequency)
            #axes[3].set_ylim((0, 30000))
            fig.tight_layout()

def plotTimeAndFreqDomain(tensor):
    for channel in range(channelCount):

        for i in range(tensor.shape[1]): #
            tensorSlice = tensor[:,i,channel]
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

                f_phs_ax = fig.add_subplot(gs[2:, 1:])
                f_phs_ax.set_title('Phase spectrum')

                f_spec_ax.magnitude_spectrum(tensorSlice, window = sp.get_window("boxcar", count), scale="dB")
                f_phs_ax.phase_spectrum(tensorSlice, window = sp.get_window("boxcar", count))

                f_re_ax.plot(np.real(tensorSlice))
                f_im_ax.plot(np.imag(tensorSlice))

                fs = signalLen
                analytic_signal = sp.hilbert(np.real(tensorSlice))
                amplitude_envelope = np.abs(analytic_signal)
                instantaneous_phase = np.unwrap(np.angle(analytic_signal))
                instantaneous_frequency = (np.diff(instantaneous_phase) / (2.0*np.pi) * fs)

                f_inst_phs_ax.plot(instantaneous_phase)
                f_inst_freq_ax.plot(instantaneous_frequency)
                #axes[3].set_ylim((0, 30000))
                #fig.tight_layout()


# channel = input("Enter channel number: ")
# channel = int(channel)
# subband = input("Enter subband number: ")
# subband = int(subband)

#plotSubbands(tensor, channel)
#plotSubband(tensor, channel, subband)
#plotSignal(tensor, channel)
plotTimeAndFreqDomain(tensor)

plt.show()