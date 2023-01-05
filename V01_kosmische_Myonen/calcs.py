try:
    import lmfit
except ImportError as e:
    print("Bitte python package: lmfit installieren!")
    quit()
from lmfit.models import ExponentialModel, GaussianModel
import numpy as np
import matplotlib.pyplot as plt
from uncertainties import ufloat, unumpy

#data loading
x0 = np.arange(512)
y = np.genfromtxt("data/daten.txt")

#data transmutation from channels to microseconds
x = 0.0216*x0 + 0.15423

mod = ExponentialModel()
pars = mod.guess(y[4::463], x=x[4::463])
out = mod.fit(y, pars, x=x)

print(out.fit_report()) #amplitude = 93.5814 +/- 1.9725 , decay = 2.5555 +/- 0.0723

del mod, pars, out

print("###------###")

t1, t2, c0 = np.genfromtxt("data/delayline.csv", unpack=True, delimiter=",")
t3 = t1-t2
t4 = np.flip(t3[1:31])
t5 = np.append(t3[0], t3[31:61])
t = np.append(t4,t5)

c1 = np.flip(c0[1:31])
c2 = np.append(c0[0], c0[31:61])
c = np.append(c1,c2)

mod = GaussianModel()
pars = mod.guess(c, x=t)
out = mod.fit(c, pars, x=t)

print(out.fit_report())

c1 = unumpy.uarray(c, np.sqrt(c))

plt.plot(t, out.best_fit, label='Gauss Fit')
plt.errorbar(t, unumpy.nominal_values(c1), yerr=unumpy.std_devs(c1), fmt='o', color='red', capsize=2, label="Messwerte mit Fehlern")
plt.xlabel('Differenz der beiden Verzögerungsleitungen in ns')
plt.ylabel('Counts pro 10 s')
plt.legend(loc='best')
plt.tight_layout()
plt.savefig("images/fwhm.pdf")