import numpy as np
import matplotlib.pyplot as plt

#1

T = np.linspace(0, 10, 100)
F_T = np.sin(0.7*np.pi*T +0.4)

plt.plot(T,F_T, label='f(t)',color='green')

#2

N = np.arange(0,11,1)
X_n = np.sin(0.7 *np.pi *N +0.4)

plt.stem(N,X_n)

#3
Ts=0.125
time = np.arange(0,10,Ts)
values= np.sin(0.7*np.pi*time +0.4)

plt.scatter(time,values, color='red')

plt.legend()
plt.grid()
plt.show()



