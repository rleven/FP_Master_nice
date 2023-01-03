import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from uncertainties import ufloat
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
x0 = np.linspace(0, 511, 512)
y0 = modelf(x0, 0.02167, 0.15423)

#Einlesen der Messdaten
x1 = np.arange(512)
y1 = np.genfromtxt("data/daten.txt")

#Anzeigen der Messdaten
plt.bar(x1, y1, label="Messung bei ca. 3 Tagen")
plt.xlabel("Markennummer")
plt.ylabel("Anzahl der Counts")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("images/messung_roh.pdf")

#Funktion für Fit
def expo(x, a, tau):
    return a*np.exp(-x/tau)

#Fit der Messdaten
plt.plot(x1, expo(x1, 88.101, 118.310), '-r')
plt.savefig("images/messung_fit.pdf")
plt.close()