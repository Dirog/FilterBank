import numpy as np

signal_file = open("./files/sine", "w")


x = np.arange(0, 511, 1)
sin = np.sin(2*np.pi*1/8*x)


for smpl in sin:
    signal_file.write("%f " % smpl)
    signal_file.write("%f " % 0.0)
