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


if (signalLen % filterLen) == 0:
    newSignalLen = signalLen
else:
    newSignalLen = signalLen + filterLen - signalLen % filterLen


count = ((newSignalLen - filterLen) // step) + 1


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
    for i in range(tensor.shape[1]): #
        tensorSlice = tensor[:,i,channel]
        if np.max(np.abs(tensorSlice)) > 2:
            fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
            fig.suptitle("subband #" + str(i) + ". Channel:" + str(channel + 1)) 
            axes[0].magnitude_spectrum(tensorSlice, window = sp.get_window("boxcar", count))
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


channel = input("Enter channel number: ")
channel = int(channel)
# subband = input("Enter subband number: ")
# subband = int(subband)

plotSubbands(tensor, channel)
#plotSubband(tensor, channel, subband)
plt.show()