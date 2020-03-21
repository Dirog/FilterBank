import numpy as np
import scipy.signal as sp 
import matplotlib.pyplot as plt

result_file = open("../python/files/result", "r")
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

tensor = np.zeros((count, fftSize, channelCount), dtype="complex128")

line = result_file.readline()
numbers = line.split()

i = 0
for c in range(channelCount):
    for n in range(count):
        for f in range(fftSize):
            tensor[n, f, c] = complex(float(numbers[2*i]), float(numbers[2*i+1]))
            i = i + 1


def plotSubbands(tensor, channel):
    for i in range(tensor.shape[1]):
        tensorSlice = tensor[:,i,channel]
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(7, 7))
        fig.suptitle("subband #" + str(i) + ". Channel:" + str(channel + 1)) 
        axes[0].magnitude_spectrum(tensorSlice, window = sp.get_window("boxcar", count))
        axes[1].phase_spectrum(tensorSlice, window = sp.get_window("boxcar", count))


channel = input("Enter channel number: ")
channel = int(channel)

plotSubbands(tensor, channel)
plt.show()

