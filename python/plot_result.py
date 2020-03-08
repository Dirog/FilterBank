import numpy as np
import matplotlib.pyplot as plt

result_file = open("./files/result", "r")

signalLen = 1024*8*2
filterLen = 512
fftSize = filterLen // 64
step = 128

count = ((signalLen // 2 - filterLen) // step) + 1

mat = np.zeros((count, fftSize), dtype="complex128")
print(count)
print(2*fftSize)

j = 0
for line in result_file:
     row = line.split()
     for i in range(fftSize):
          mat[j, i] = complex(float(row[2*i]), float(row[2*i+1]))
     j = j + 1

def plotSubbands(fft_matrix, step = 1):
    for n in range(0, fft_matrix.shape[1], step):
        plt.figure()
        plt.stem(np.abs(np.fft.ifft(fft_matrix[:,n])), use_line_collection="true")
        plt.ylim((0, 1)) 
        #plt.stem(np.real(fft_matrix[:,n]), use_line_collection="true")
        #plt.ylim((-1, 1)) 
        plt.title("subband #" + str(n))


plotSubbands(mat, 1)
plt.show()

