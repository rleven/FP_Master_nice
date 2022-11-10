import numpy as np
import matplotlib.pyplot as plt

#funktionen

def B_feld(N, I , R):
    return 9*10**(-3) * (N*I)/R #return ist in Gaus


#Messwerte einlesen
f, Isweep_peak1, I_hor_peak1, Isweep_peak2, I_hor_peak2 = np.genfromtxt('Data/rf_modulation.txt', comments='#', unpack=True, delimiter=', ')


#Als erstes Magnetfeld staerke der vertikal spule berechnen
#1 Umdrehung = 0.1A

I_ver = 0.1 * 2.25 # vertikal spulen Strom vertikal
B_ver = B_feld(20, I_ver, 11.735*10**(-2))

print('Vertikales B-Feld zur komopensation des Erdmagnetfelds: ', B_ver)



Bsweep_peak1 = B_feld(11, Isweep_peak1, 0.1639) # Bfeld der Sweep Spule bei Peak 1
Bsweep_peak2 = B_feld(11, Isweep_peak2, 0.1639) # Bfeld der Sweep Spule bei Peak 2
Bhor_peak1 = B_feld(154, I_hor_peak1, 0.1579) # Bfeld der Horizontal Spule bei Peak 1
Bhor_peak2 = B_feld(154, I_hor_peak2, 0.1579) # Bfeld der Horizontal Spule bei Peak 2

#Da die Vertikale Komponente der Spulen das Erdmagnetfeld kompensiert ist für spätere rechungen noch die horizontal/sweep komponente von belang

plt.figure()

plt.plot(f,Bsweep_peak1+Bhor_peak1, label='Messwerte 1 Peak')
plt.plot(f, Bsweep_peak2+Bhor_peak2, label='Messwerte 2 Peak')
plt.legend()
plt.xlabel('Horizontales Magnetfeld ' + r'$B / G$')
plt.ylabel('RF-Spulen Frequenz ' + r'$f/Hz$')
plt.savefig('content/plots/landefaktor.pdf')