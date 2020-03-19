import numpy as np
import matplotlib.pyplot as plt

result_file = open("./python/files/result", "r")

signalLen = 1024*256*2
filterLen = 1024
fftSize = filterLen // 128
step = 128
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
          



def plotSubbandsAR(tensor, channel):
    for i in range(tensor.shape[1]):
        tensorSlice = tensor[:,i,channel]
        plt.figure()
        plt.stem(np.abs(np.fft.ifft(tensorSlice)), use_line_collection="true")
        #plt.plot(np.real(tensorSlice))
        plt.ylim((0, 1)) 
        plt.title("subband #" + str(i) + ". Channel:" + str(channel))
       
def plotSubbandsPR(tensor, channel):
    for i in range(tensor.shape[1]):
        tensorSlice = tensor[:,i,channel]
        ifft = np.fft.ifft(tensorSlice)
        ifft[np.abs(ifft) < 1e-6] = 0
        plt.figure()
        plt.stem(np.angle(ifft), use_line_collection="true")
        plt.ylim((-np.pi, np.pi)) 
        plt.title("subband #" + str(i) + ". Channel:" + str(channel))


channel = input("Enter channel number: ")
channel = int(channel)

plotSubbandsAR(tensor, channel)

plt.show()

