import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft

# Параметри
N = 64
n = np.arange(N)

# Дефинирање сигнали
x = np.zeros(N)
x[0:8]=1

y= np.zeros(N)
y[56:64]=1

# DFT
X = fft(x)
Y = fft(y)

# Амплитуда и фаза
amp_X = np.abs(X)
amp_Y = np.abs(Y)

phase_X = np.angle(X)
phase_Y = np.angle(Y)

# Плотови
fig, axs = plt.subplots(2, 2, figsize=(12, 8))

# Амплитудни спектри
axs[0, 0].stem(n, amp_X, basefmt=" ")
axs[0, 0].set_title("Амплитуден спектар на x[n]")
axs[0, 1].stem(n, amp_Y, basefmt=" ")
axs[0, 1].set_title("Амплитуден спектар на y[n]")

# Фазни спектри
axs[1, 0].stem(n, phase_X, basefmt=" ")
axs[1, 0].set_title("Фазен спектар на x[n]")
axs[1, 1].stem(n, phase_Y, basefmt=" ")
axs[1, 1].set_title("Фазен спектар на y[n]")

plt.tight_layout()
plt.show()


#2


# Сигнали
x = np.sin(np.pi * n / 16) + 2 * np.cos(np.pi * n / 8)
y = np.sin(np.pi * n / 16 + np.pi / 3) + 2 * np.cos(np.pi * n / 8)

# DFT
X = fft(x)
Y = fft(y)

# Амплитуда и фаза
amp_X = np.abs(X)
amp_Y = np.abs(Y)

phase_X[amp_X < 1e-10] = 0
phase_Y[amp_Y < 1e-10] = 0


# Временски домен
plt.figure(figsize=(12, 4))
plt.plot(n, x, label='x[n]')
plt.plot(n, y, label='y[n]', linestyle='--')
plt.title("Временски сигнали x[n] и y[n]")
plt.xlabel("n")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Амплитуден спектар
plt.figure(figsize=(12, 4))
plt.stem(n, amp_X, basefmt=" ", label="|X[k]|")
plt.stem(n, amp_Y, basefmt=" ", linefmt='r-', markerfmt='ro', label="|Y[k]|")
plt.title("Амплитуден спектар")
plt.xlabel("k")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Фазен спектар
plt.figure(figsize=(12, 4))
plt.stem(n, phase_X, basefmt=" ", label="∠X[k]")
plt.stem(n, phase_Y, basefmt=" ", linefmt='r-', markerfmt='ro', label="∠Y[k]")
plt.title("Фазен спектар")
plt.xlabel("k")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()


#3
N = 64
n = np.arange(N)

# Сигнали
x = np.array([1 if i % 2 == 0 else 0 for i in n])
y = np.cos(np.pi * n)  # (-1)^n
z = np.ones(N)

# Проверка: x = (z + y) / 2
x_from_yz = (z + y) / 2
assert np.allclose(x, x_from_yz), "x[n] ≠ (z[n]+y[n])/2"

# DFT
X = fft(x)
Y = fft(y)
Z = fft(z)

# Проверка: X = (Z + Y) / 2
X_from_YZ = (Z + Y) / 2
assert np.allclose(X, X_from_YZ), "X[k] ≠ (Z[k]+Y[k])/2"

# Временски домен
plt.figure(figsize=(12, 4))
plt.stem(n, x, linefmt='b-', markerfmt='bo', basefmt=" ", label='x[n]')
plt.stem(n, y, linefmt='r-', markerfmt='ro', basefmt=" ", label='y[n]')
plt.stem(n, z, linefmt='g-', markerfmt='go', basefmt=" ", label='z[n]')
plt.title("Сигнали x[n], y[n], z[n]")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Амплитуден спектар
plt.figure(figsize=(12, 4))
plt.stem(n, np.abs(X), linefmt='b-', markerfmt='bo', basefmt=" ", label='|X[k]|')
plt.stem(n, np.abs(Y), linefmt='r-', markerfmt='ro', basefmt=" ", label='|Y[k]|')
plt.stem(n, np.abs(Z), linefmt='g-', markerfmt='go', basefmt=" ", label='|Z[k]|')
plt.title("Амплитудни спектри")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Фазен спектар
plt.figure(figsize=(12, 4))
plt.stem(n, np.angle(X), linefmt='b-', markerfmt='bo', basefmt=" ", label='∠X[k]')
plt.stem(n, np.angle(Y), linefmt='r-', markerfmt='ro', basefmt=" ", label='∠Y[k]')
plt.stem(n, np.angle(Z), linefmt='g-', markerfmt='go', basefmt=" ", label='∠Z[k]')
plt.title("Фазни спектри")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()