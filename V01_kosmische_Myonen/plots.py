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

yerrlow = np.sqrt(y1[4:463])+3.423

for i in range(len(y1[4:463])):
    if np.sqrt(y1[i+4]) <= 3.423:
        yerrlow[i] = y1[i+4]

#Fit der Messdaten
plt.errorbar(y0[4:463], y1[4:463], yerr=[yerrlow, np.sqrt(y1[4:463])], color='lightblue', ls='none', linewidth=1, capsize=1, label='Unsicherheit Messung')
plt.plot(y0, y1, '.b', ms=1, label="Messung bei ca. 3 Tagen")
plt.plot(y0, unumpy.nominal_values(y2), '-r', label="Exponentieller Fit")
plt.fill_between(y0, unumpy.nominal_values(y2) - unumpy.std_devs(y2), unumpy.nominal_values(y2) + unumpy.std_devs(y2), color='red', alpha=0.5, label='Unsicherheit Fit')
plt.xlabel(r"Zerfallszeit in $\mu s$")
plt.ylabel("Anzahl der Counts")
plt.legend(loc="best")
plt.tight_layout()
plt.savefig("images/messung_fit.pdf")
plt.close()


def plateau(x, a, x0, sig, n):
    return a*(np.exp(-(np.abs(x - x0)/sig)**(2*n)))

def plateau2(x, a, x0, sig, n):
    return a*(unumpy.exp(-(unumpy.fabs(x - x0)/sig)**(2*n)))

t1, t2, c0 = np.genfromtxt("data/delayline.csv", unpack=True, delimiter=",")
t3 = t1-t2
t4 = np.flip(t3[1:31])
t5 = np.append(t3[0], t3[31:61])
t = np.append(t4,t5)

c1 = np.flip(c0[1:31])
c2 = np.append(c0[0], c0[31:61])
c = np.append(c1,c2)

popt, pvoc = curve_fit(plateau, t, c, p0=[100, 1, 50, 4])

print(popt)

c1 = unumpy.uarray(c, np.sqrt(c))

plt.plot(t, plateau(t, 215.914, 1.617, 14.843, 1.675), label='Plateau Fit')
plt.errorbar(t, unumpy.nominal_values(c1), yerr=unumpy.std_devs(c1), fmt='o', color='red', capsize=2, label="Messwerte mit Fehlern")
plt.xlabel('Differenz der beiden Verzögerungsleitungen in ns')
plt.ylabel('Counts pro 10 s')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("images/fwhm.pdf")

dom = np.linspace(-15, 17, 10000)
udom = unumpy.uarray(dom, 0.001)
y = plateau2(udom, 215.914, 1.617, 14.843, 1.675)
arr = np.array([udom, y])
spec = (np.max(arr[1][:])-np.min(arr[1][:]))/2
print(spec)
shit = np.isclose(unumpy.nominal_values(arr[1]), unumpy.nominal_values(spec), rtol=0.001)
for i in range(len(shit)):
    if shit[i] == True:
        print(i, arr[1][i])
    else:
        continue

print(arr[0][614] - arr[0][9771])