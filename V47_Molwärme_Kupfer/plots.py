import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat, unumpy

#data aqcusition
t, shield, probe, U, I = np.genfromtxt('Data/data.txt', unpack=True, delimiter = ',', comments = '#')

#defining nessecary functions
def temperatur(R):
    return 0.00134*R*R + 2.296*R - 243.02 + 273.15

def druckwärme(t, U, I, dT):
    return U*I*t*0.063546/(dT*0.342)

def volwärme(cp, T):
    return cp - 9*modelf(T, 3.46524982, 56.0189805, -2.29906327)**2*137.8e9*0.063546/8960*T*10e-12

def modelf(x, a, b, c):
    return (a*unumpy.log(x-b) + c)

#creating standard deviation for data
uprobe = unumpy.uarray(probe, np.full(len(probe), 0.1))
ushield = unumpy.uarray(shield, np.full(len(shield), 0.1))

uU = unumpy.uarray(U, np.full(len(U), 0.01))
uI = unumpy.uarray(I, np.full(len(I), 0.0001))

uTmp = temperatur(uprobe)

duT = unumpy.uarray(np.zeros(len(uTmp)-1), np.zeros(len(uTmp)-1))
for i in range(len(duT)):
    duT[i] = uTmp[i+1] - uTmp[0]

#plot for raw temperature
plt.plot(t, unumpy.nominal_values(uTmp), '-b', label='temperature of Cu probe')
plt.xlabel('t in s')
plt.ylabel('Temperature in K')
plt.grid()
plt.legend(loc='best')
plt.savefig('content/plots/temperature.pdf')
plt.close()

#plot for temerature deviation
plt.fill_between(t, unumpy.nominal_values(uTmp) - unumpy.std_devs(uTmp), unumpy.nominal_values(uTmp) + unumpy.std_devs(uTmp), alpha=0.3, label='uncertainty')
plt.xlabel('t in s')
plt.ylabel('Temperature in K')
plt.grid()
plt.legend(loc='best')
plt.savefig('content/plots/utemperature.pdf')
plt.close()

#calculating pressure heat capacity
druckedy = druckwärme(t[1::], uU[1::], uI[1::], duT)

#plot for heat capacity with deviation
plt.plot(unumpy.nominal_values(uTmp[1::]), unumpy.nominal_values(druckedy), '-r', label='heat capacity')
plt.fill_between(unumpy.nominal_values(uTmp[1::]), unumpy.nominal_values(druckedy) - unumpy.std_devs(druckedy), unumpy.nominal_values(druckedy) + unumpy.std_devs(druckedy), color='red', alpha=0.3, label='uncertainty')
plt.xlabel('T in K')
plt.ylabel(r'$C_P$ in $\frac{J}{K\cdot mol}$')
plt.grid()
plt.title('Heat capacity at constant pressure')
plt.legend(loc='best')
plt.savefig('content/plots/ucp.pdf')
plt.close()

#defining values and fit for thermal expansion
x = np.linspace(70, 300, 24)
y = np.array([7, 8.5, 9.75, 10.7, 11.5, 12.1, 12.65, 13.15, 13.6, 13.9, 14.25, 14.5, 14.75, 14.95, 15.2, 15.4, 15.6, 15.75, 15.9, 16.1, 16.25, 16.35, 16.5, 16.65])

#plot for thermal expansion fit
plt.plot(x, y, '-k', label=r'coefficient of thermal expansion $\alpha$')
plt.plot(x, modelf(x, 3.46524982, 56.0189805, -2.29906327), '--', color='orange', label='fit')
plt.xlabel('T in K')
plt.ylabel(r'$\alpha$ in $10^{-6}K^{-1}$')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig('content/plots/coefficient.pdf')
plt.close()

#calculating volume heat capacity
cv = volwärme(druckedy, uTmp[1::])

#plot for volume heat capacity
plt.plot(unumpy.nominal_values(uTmp[1::]), unumpy.nominal_values(cv), '-r', label='heat capacity')
plt.fill_between(unumpy.nominal_values(uTmp[1::]), unumpy.nominal_values(cv) - unumpy.std_devs(cv), unumpy.nominal_values(cv) + unumpy.std_devs(cv), color='red', alpha=0.3, label='uncertainty')
plt.xlabel('T in K')
plt.ylabel(r'$C_V$ in $\frac{J}{K\cdot mol}$')
plt.grid()
plt.title('Heat capacity at constant volume')
plt.legend(loc='best')
plt.savefig('content/plots/ucv.pdf')
plt.close()

#painfully including the matrix
debye_matrix = np.array([[24.9430, 24.9310, 24.8930, 24.8310, 24.7450, 24.6340, 24.5000, 24.3430, 24.1630, 23.9610],
                         [23.7390, 23.4970, 23.2360, 22.9560, 22.6600, 22.3480, 22.0210, 21.6800, 21.3270, 20.9630],
                         [20.5880, 20.2050, 19.8140, 19.4160, 19.0120, 18.6040, 18.1920, 17.7780, 17.3630, 16.9470],
                         [16.5310, 16.1170, 15.7040, 15.2940, 14.8870, 14.4840, 14.0860, 13.6930, 13.3050, 12.9230],
                         [12.5480, 12.1790, 11.8170, 11.4620, 11.1150, 10.7750, 10.4440, 10.1190, 9.8030, 9.4950],
                         [9.1950, 8.9030, 8.6190, 8.3420, 8.0740, 7.8140, 7.5610, 7.3160, 7.0780, 6.8480],
                         [6.6250, 6.4090, 6.2000, 5.9980, 5.8030, 5.6140, 5.4310, 5.2550, 5.0840, 4.9195],
                         [4.7606, 4.6071, 4.4590, 4.3160, 4.1781, 4.0450, 3.9166, 3.7927, 3.6732, 3.5580],
                         [3.4468, 3.3396, 3.2362, 3.1365, 3.0403, 2.9476, 2.8581, 2.7718, 2.6886, 2.6083],
                         [2.5309, 2.4562, 2.3841, 2.3146, 2.2475, 2.1828, 2.1203, 2.0599, 2.0017, 1.9455],
                         [1.8912, 1.8388, 1.7882, 1.7393, 1.6920, 1.6464, 1.6022, 1.5596, 1.5184, 1.4785],
                         [1.4400, 1.4027, 1.3667, 1.3318, 1.2980, 1.2654, 1.2337, 1.2031, 1.1735, 1.1448],
                         [1.1170, 1.0900, 1.0639, 1.0386, 1.0141, 0.9903, 0.9672, 0.9449, 0.9232, 0.9021],
                         [0.8817, 0.8618, 0.8426, 0.8239, 0.8058, 0.7881, 0.7710, 0.7544, 0.7382, 0.7225],
                         [0.7072, 0.6923, 0.6779, 0.6638, 0.6502, 0.6368, 0.6239, 0.6113, 0.5990, 0.5871],
                         [0.5755, 0.5641, 0.5531, 0.5424, 0.5319, 0.5210, 0.5117, 0.5020, 0.4926, 0.4834]])

#writing the algorithm for automatic recognition of debye number
cap = unumpy.nominal_values(cv) #only until cap[61]

Deb = np.zeros(62)  #stores the debye numbers after the for loop

for i in range(62):
    l = 0
    count = np.zeros(40)
    if cap[i] >= 12.923:
        for j in range(4):
            for k in range(10):
                count[l] = np.abs(debye_matrix[j, k] - cap[i])
                l+=1
    elif cap[i] >= 3.558:
        for j in range(4, 8):
            for k in range(10):
                count[l] = np.abs(debye_matrix[j, k] - cap[i])
                l+=1
    
    val = int(np.where(count == np.min(count))[0])   #gives back the index of the lowest difference
    
    if cap[i] >= 12.923:
        Deb[i] = val/10.
    elif cap[i] >= 3.558:
        Deb[i] = 4 + val/10.
    
    del count, l

print(Deb)

plt.plot(unumpy.nominal_values(uTmp[1:63]), Deb, '-g')
plt.xlabel('T in K')
plt.ylabel(r'$\frac{\theta_D}{T}$')
plt.grid()
plt.title(r'Behaviour of $\frac{\theta_D}{T}$')
plt.tight_layout()
plt.savefig('content/plots/debye.pdf')
plt.close()