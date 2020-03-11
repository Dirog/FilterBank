import numpy as np
import matplotlib.pyplot as plt

result_file = open("../python/files/result", "r")

signalLen = 1024*8*2
filterLen = 128
fftSize = filterLen // 16
step = 32
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

print(i)
          





def plotSubbands(tensor, channel):
    for i in range(tensor.shape[1]):
        tensorSlice = tensor[:,i,channel]
        plt.figure()
        plt.stem(np.abs(np.fft.ifft(tensorSlice)), use_line_collection="true")
        plt.ylim((0, 1)) 
        #plt.stem(np.real(fft_matrix[:,n]), use_line_collection="true")
        #plt.ylim((-1, 1)) 
        plt.title("subband #" + str(i) + ". Channel:" + str(channel))


channel = input("Enter channel number: ")
#subband = input("Enter subband number: ")
channel = int(channel)
#subband = int(subband)
plotSubbands(tensor, channel)

plt.show()
