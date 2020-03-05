import numpy as np
import matplotlib.pyplot as plt

result_file = open("./files/result", "r")

fftSize = 256
count = 95
mat = np.zeros((count, fftSize), dtype="complex128")

j = 0
for line in result_file:
     row = line.split()
     for i in range(fftSize):
          mat[j, i] = complex(float(row[2*i]), float(row[2*i+1]))
     j = j + 1

plt.matshow(np.abs(np.transpose(mat)), origin='lower')
plt.show()

