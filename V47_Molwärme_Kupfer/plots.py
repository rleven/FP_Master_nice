import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat

t, shield, probe, U, I = np.genfromtxt('Data/data.txt', unpack=True, delimiter = ',', comments = '#')

def temperatur(R):
    return 0.00134*R*R + 2.296*R -243.02

def druckwärme(t, U, I, dT):
    return U*I*t*0.063546/(dT*0.342)

Tmp = temperatur(probe)
dT = np.zeros(len(Tmp)-1)
for i in range(len(dT)):
    dT[i] = Tmp[i+1] - Tmp[0]

#plt.plot(t, shield, '-r', label='shield')
plt.plot(t[1::], druckwärme(t[1::], U[1::], I[1::], dT), '-b', label='heat capacity')
plt.xlabel('t in s')
plt.ylabel(r'$C_P$')
plt.legend(loc='best')
plt.show()