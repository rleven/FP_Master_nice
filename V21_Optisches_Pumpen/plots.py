import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
from scipy.constants import mu_0 , h , elementary_charge , electron_mass, hbar , eV
mu_B = elementary_charge*hbar/(2*electron_mass)
E_Hyper_1 = 4.53*10**(-24) #J
E_Hyper_2 = 2.01*10**(-24) #J
#funktionen

def B_feld(N, I , R):
    return 9*10**(-7) * (N*I)/R #return ist in Gaus

def gerade(x,a,b):
    return a*x+b

def lande(a):
    return h/(mu_B*a)

def I(landefaktor):
    return 1/landefaktor-0.5

def delta_E(B,g_F,M_F,E_Hyperfein):
    return g_F*mu_B*B+(1-2*M_F)*(g_F*mu_B*B)**2/E_Hyperfein


#Messwerte einlesen
f, Isweep_peak1, I_hor_peak1, Isweep_peak2, I_hor_peak2 = np.genfromtxt('Data/rf_modulation.txt', comments='#', unpack=True, delimiter=', ')

#f von MHz zu Hz
f = f*10**(6)

#Als erstes Magnetfeld staerke der vertikal spule berechnen
#1 Umdrehung = 0.1A

I_ver = 0.1 * 2.25 # vertikal spulen Strom vertikal
B_ver = B_feld(20, I_ver, 11.735*10**(-2))

#hor spule umdrehung = 0.3A
I_hor_peak1 = I_hor_peak1*0.3
I_hor_peak2 = I_hor_peak2*0.3

#sweep spule umdrehung = 0.1 A
Isweep_peak1 = Isweep_peak1*0.1
Isweep_peak2 = Isweep_peak2*0.1

print('Vertikales B-Feld zur komopensation des Erdmagnetfelds: ', B_ver)


#Die anderen B felder berechnen
Bsweep_peak1 = B_feld(11, Isweep_peak1, 0.1639) # Bfeld der Sweep Spule bei Peak 1 in T
Bsweep_peak2 = B_feld(11, Isweep_peak2, 0.1639) # Bfeld der Sweep Spule bei Peak 2
Bhor_peak1 = B_feld(154, I_hor_peak1, 0.1579) # Bfeld der Horizontal Spule bei Peak 1
Bhor_peak2 = B_feld(154, I_hor_peak2, 0.1579) # Bfeld der Horizontal Spule bei Peak 2


#ausgleichsrechung
params1, cov1 = curve_fit(gerade, f, Bsweep_peak1+ Bhor_peak1)
params2, cov2 = curve_fit(gerade, f, Bsweep_peak2+ Bhor_peak2)
x = np.linspace(np.min(f), np.max(f), 1000)

lande1 = ufloat(params1[0], cov1[0][0]**0.5)
lande2 = ufloat(params2[0], cov2[0][0]**0.5)

lande1 = lande(lande1)
lande2 = lande(lande2)

B_1 = Bsweep_peak1+Bhor_peak1
B_2 = Bsweep_peak2+Bhor_peak2

del_E1 = delta_E(B_1[-1],lande1,2,E_Hyper_1)
del_E2 = delta_E(B_2[-1],lande2,3,E_Hyper_2)

del_E1_eV = del_E1/eV*10**9
del_E2_eV = del_E2/eV*10**9

print('Landefaktor Isotop 1: ',lande1, 'Erdmagnetfeld horizontal: ', ufloat(params1[1], cov1[1][1]**0.5))
print('Landefaktor Isotop 2: ', lande2, 'Erdmagnetfeld horizontal: ', ufloat(params2[1], cov2[1][1]**0.5))
print('Kernspin I Iostop 1: ', I(lande1))
print('Kernspin I Iostop 2: ', I(lande2))
print('Quadratischer zeeman 1: ', del_E1_eV, 'zu Magnetfeld: ', B_1[-1])
print('Quadratischer zeeman 2: ', del_E2_eV, 'zu Magnetfeld: ', B_2[-1])
print('Abweichung der beiden Erdmagnetfeldwerte: ', abs(ufloat(params1[1], cov1[1][1]**0.5)-ufloat(params2[1], cov2[1][1]**0.5)/ufloat(params2[1], cov2[1][1]**0.5)))

#Da die Vertikale Komponente der Spulen das Erdmagnetfeld kompensiert ist für spätere rechungen noch die horizontal/sweep komponente von belang

plt.figure()

plt.plot(f/10**3,Bsweep_peak1+Bhor_peak1,'x', label='Messwerte Isotop 1')
plt.plot(x/10**3, gerade(x, params1[0], params1[1]), label='Ausgleichsgerade Isotop 1')
plt.plot(f/10**3, Bsweep_peak2+Bhor_peak2,'kx', label='Messwerte Isotop 2')
plt.plot(x/10**3, gerade(x, params2[0], params2[1]), label='Ausgleichsgerade Isotop 2')
plt.legend()
plt.ylabel('Horizontales Magnetfeld ' + r'$B / G$')
plt.xlabel('RF-Spulen Frequenz ' + r'$f/kHz$')
plt.tight_layout()
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

t1 = t1/num_per1 *10**-3
t2 = t2/num_per2 *10**-3

#########

params3, cov3 = curve_fit(hyp, ampli1, t1, p0=(1,1))
params4, cov4 = curve_fit(hyp, ampli2, t2, p0=(1,1))

x = np.linspace(np.min(ampli1), np.max(ampli1), 1000)

print('Fitparameter 1 a:', ufloat(params3[0], cov3[0][0]**0.5),'b: ', ufloat(params3[1], cov3[1][1]**0.5))
print('Fitparameter 2 a:', ufloat(params4[0], cov4[0][0]**0.5),'b: ', ufloat(params4[1], cov4[1][1]**0.5))
print('Gefragt wert b/b: ', ufloat(params4[1], cov4[1][1]**0.5)/ufloat(params3[1], cov3[1][1]**0.5))

plt.figure()


plt.plot(ampli1,t1*10**3, 'x',label='Messwerte Isotop 1')
plt.plot(x, hyp(x, params3[0], params3[1])*10**3,'--',alpha=0.8 ,label='Ausgleichsgerade Isotop 1')
plt.plot(ampli2,t2*10**3, 'kx',label='Messwerte Isotop 2')
plt.plot(x, hyp(x, params4[0], params4[1])*10**3,'--',alpha=0.8, label='Ausgleichsgerade Isotop 2')
plt.legend()
plt.ylabel('Periodendauer ' + r'$T / ms$')
plt.xlabel('RF-Amplitude ' + r'$/V$')
plt.savefig('content/plots/periodendauer.pdf')
