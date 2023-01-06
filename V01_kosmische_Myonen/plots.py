import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat, unumpy
from scipy.constants import mu_0 , h , elementary_charge , electron_mass, hbar , eV

#Einlesen der Daten für Marker-Zerfallszeit-Abhängigkeit
x = np.genfromtxt("data/marker.csv", unpack=True, delimiter=",", usecols=0)
t = np.genfromtxt("data/marker.csv", unpack=True, delimiter=",", usecols=2)

#Genutztes Model für Zerfallszeit-Fit
def modelf(x, a, b):
    return a*x + b

#Fit
popt, pvoc = curve_fit(modelf, x, t, p0=[0.02, 0.3])

#Neue Kurve nach ausgerechnetem Fit
x0 = np.arange(512)
y0 = modelf(x0, 0.02167, 0.15423)
#print(y0[0])
plt.plot(x, t, '.b', label="Kalibrationsdaten")
plt.xlabel("Kanalnummer")
plt.ylabel(r"Pulsabstand in $\mu s$")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("images/linear.pdf")
plt.close()

#Einlesen der Messdaten
y1 = np.genfromtxt("data/daten.txt")

#Anzeigen der Messdaten
plt.bar(x0, y1, label="Messung bei ca. 3 Tagen")
plt.xlabel("Kanalnummer")
plt.ylabel("Anzahl der Counts")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("images/messung_roh.pdf")
plt.close()

#Funktion für Fit
def expo(x, a, tau):
    return a*unumpy.exp(-x/tau)

y2 = expo(y0, ufloat(93.6524, 1.1433), ufloat(2.5454, 0.0417))

#Fit der Messdaten
plt.plot(y0, y1, label="Messung bei ca. 3 Tagen")
plt.plot(y0, unumpy.nominal_values(y2), '-r', label="Exponentieller Fit")
plt.fill_between(y0, y1 - np.sqrt(y1), y1 + np.sqrt(y1), color='blue', alpha=0.3, label='Unsicherheit Messung')
plt.fill_between(y0, unumpy.nominal_values(y2) - unumpy.std_devs(y2), unumpy.nominal_values(y2) + unumpy.std_devs(y2), color='red', alpha=0.5, label='Unsicherheit Fit')
plt.xlabel(r"Zerfallszeit in $\mu s$")
plt.ylabel("Anzahl der Counts")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("images/messung_fit.pdf")
plt.close()