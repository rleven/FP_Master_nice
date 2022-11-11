import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

#funktionen

def B_feld(N, I , R):
    return 9*10**(-3) * (N*I)/R #return ist in Gaus

def gerade(x,a,b):
    return a*x+b

#Messwerte einlesen
f, Isweep_peak1, I_hor_peak1, Isweep_peak2, I_hor_peak2 = np.genfromtxt('Data/rf_modulation.txt', comments='#', unpack=True, delimiter=', ')


#Als erstes Magnetfeld staerke der vertikal spule berechnen
#1 Umdrehung = 0.1A

I_ver = 0.1 * 2.25 # vertikal spulen Strom vertikal
B_ver = B_feld(20, I_ver, 11.735*10**(-2))

print('Vertikales B-Feld zur komopensation des Erdmagnetfelds: ', B_ver)


#Die anderen B felder berechnen
Bsweep_peak1 = B_feld(11, Isweep_peak1, 0.1639) # Bfeld der Sweep Spule bei Peak 1
Bsweep_peak2 = B_feld(11, Isweep_peak2, 0.1639) # Bfeld der Sweep Spule bei Peak 2
Bhor_peak1 = B_feld(154, I_hor_peak1, 0.1579) # Bfeld der Horizontal Spule bei Peak 1
Bhor_peak2 = B_feld(154, I_hor_peak2, 0.1579) # Bfeld der Horizontal Spule bei Peak 2

#ausgleichsrechung
params1, cov1 = curve_fit(gerade, f, Bsweep_peak1+ Bhor_peak1)
params2, cov2 = curve_fit(gerade, f, Bsweep_peak2+ Bhor_peak2)
x = np.linspace(np.min(f), np.max(f), 1000)

print('Landefaktor Isotop 1: ', params1[0], 'Erdmagnetfeld horizontal: ', params1[1])
print('Landefaktor Isotop 2: ', params2[0], 'Erdmagnetfeld horizontal: ', params2[1])
print('Abweichung der beiden Erdmagnetfeldwerte: ', abs(params1[1]-params2[1])/params2[1])

#Da die Vertikale Komponente der Spulen das Erdmagnetfeld kompensiert ist für spätere rechungen noch die horizontal/sweep komponente von belang

plt.figure()

plt.plot(f,Bsweep_peak1+Bhor_peak1, label='Messwerte Isotop 1')
plt.plot(x, gerade(x, params1[0], params1[1]), label='Ausgleichsgerade Isotop 1')
plt.plot(f, Bsweep_peak2+Bhor_peak2, label='Messwerte Isotop 2')
plt.plot(x, gerade(x, params2[0], params2[1]), label='Ausgleichsgerade Isotop 2')
plt.legend()
plt.ylabel('Horizontales Magnetfeld ' + r'$B / G$')
plt.xlabel('RF-Spulen Frequenz ' + r'$f/Hz$')
plt.savefig('content/plots/landefaktor.pdf')

plt.close()

##############################################################################
#
# Jetzt berechung der Periodendauer
#
##############################################################################

def hyp(x, a, b):
    return a+ b/(x)

#

ampli1, num_per1, t1, fil = np.genfromtxt('Data/rabi_osc1.txt', comments='#', unpack=True, delimiter=',')
ampli2, num_per2, t2 = np.genfromtxt('Data/rabi-osc2.txt', comments='#', unpack=True, delimiter=',')


#perioden dauer einer periode berechnen

t1 = t1/num_per1
t2 = t2/num_per2

#########

params3, cov3 = curve_fit(hyp, ampli1, t1, p0=(1,1))
params4, cov4 = curve_fit(hyp, ampli2, t2, p0=(1,1))

x = np.linspace(np.min(ampli1), np.max(ampli1), 1000)

print('Gefragt wert b/b: ', params4[1]/params3[1])

plt.figure()


plt.plot(ampli1,t1, label='Messwerte Isotop 1')
plt.plot(x, hyp(x, params3[0], params3[1]),'--',alpha=0.8 ,label='Ausgleichsgerade Isotop 1')
plt.plot(ampli2,t2, label='Messwerte Isotop 2')
plt.plot(x, hyp(x, params4[0], params4[1]),'--',alpha=0.8, label='Ausgleichsgerade Isotop 2')
plt.legend()
plt.ylabel('Periodendauer ' + r'$T / s$')
plt.xlabel('RF-Amplitude ' + r'$/V$')
plt.savefig('content/plots/periodendauer.pdf')
