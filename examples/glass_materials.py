import numpy as np, sys, matplotlib.pyplot as plt, matplotlib.gridspec as gridspec

sys.path.append("../")
from src import *

BK7 = Glass_NBK7()
# plot glass refractive index from 400nm to 1500nm
wl_m = np.linspace(400e-9, 1500e-9, 1000)  # wavelength in meters
n_BK7 = BK7.n(wl_m)  # refractive index of BK
plt.plot(wl_m * 1e9, n_BK7, label="BK7")  # convert wavelength to nm
plt.grid()
plt.xlabel("Wavelength (nm)")
plt.ylabel("Refractive Index")
plt.title("Refractive Index of BK7 Glass")
plt.show()
