import numpy as np
import matplotlib.pyplot as plt

result_file = open("./python/files/result", "r")

signalLen = 2048*2
filterLen = 128
fftSize = filterLen // 32
step = 4
channelCount = 3

count = ((signalLen // 2 - filterLen) // step) + 1

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


def plotSubbandsAR(tensor, channel):
    for i in range(tensor.shape[1]):
        tensorSlice = tensor[:,i,channel]
        plt.figure()
        plt.plot(np.abs(np.fft.ifftshift(np.fft.ifft(tensorSlice))))
        #plt.plot(np.real(tensorSlice))
        plt.ylim((0, 0.5)) 
        plt.title("subband #" + str(i) + ". Channel:" + str(channel + 1))
        #plt.savefig('channel_%d_subband_%d.png' % ((channel + 1), i))

def plotSubbandsPR(tensor, channel):
    for i in range(tensor.shape[1]):
        tensorSlice = tensor[:,i,channel]
        ifft = np.fft.ifftshift(np.fft.ifft(tensorSlice))
        #ifft[np.abs(ifft) < 1e-5] = 0
        plt.figure()
        plt.plot(np.angle(ifft))
        plt.ylim((-np.pi, np.pi)) 
        plt.title("subband #" + str(i) + ". Channel:" + str(channel + 1))
        #plt.savefig('channel_%d_subband_phase_%d.png' % ((channel + 1), i))


channel = input("Enter channel number: ")
channel = int(channel)

plotSubbandsAR(tensor, channel)
plt.show()

