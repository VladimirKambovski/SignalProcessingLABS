import numpy as np
import matplotlib.pyplot as plt
import cmath

from scipy.fft import fft

def dft_matrix(N):
    #generira dft matrica
    W = np.zeros((N, N), dtype=complex)
    for n in range(N):
        for k in range(N):
            W[n, k] = cmath.exp(-2j * np.pi * n * k / N)
    return W


def dft(x):
    #presmetuva dft koristejki transformatorska matrica
    N = len(x)
    W = dft_matrix(N)
    return np.dot(W, x)

#signal

N=64
n=np.arange(N)
Omega = (2.0 * np.pi / 64) * 2
Fi = np.pi / 6
x = 2 * np.cos(Omega * n + Fi) + np.sin(Omega * 3 * n) + np.cos(Omega * 5 * n + np.pi/3)

#presmetka na dft

X_dft = dft(x)
X_fft = fft(x) #od scippy

amp_dft = np.abs(X_dft)
amp_fft = np.abs(X_fft)

phase_dft = np.angle(X_dft)
phase_fft = np.angle(X_fft)

phase_dft[amp_dft < 1e-10] = 0
phase_fft[amp_fft < 1e-10] = 0

#PRIKAZ

plt.figure() #AMPLITUDA
plt.stem(n, abs(X_dft), linefmt='b', markerfmt='bo', basefmt='r-', label="DFT")
plt.stem(n, abs(X_fft), linefmt='g', markerfmt='go', basefmt='r-', label="FFT")
plt.legend()

plt.figure() #FAZA
plt.stem(n, phase_dft, linefmt='b', markerfmt='bo', basefmt='r-', label="DFT")
plt.stem(n, phase_fft, linefmt='g', markerfmt='go', basefmt='r-', label="FFT")
plt.legend()
plt.show()
