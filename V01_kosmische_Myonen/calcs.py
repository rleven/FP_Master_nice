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
pars = mod.guess(y[4:463], x=x[4:463])
out = mod.fit(y, pars, x=x)

print(out.fit_report()) #amplitude = 93.5814 +/- 1.9725 , decay = 2.5555 +/- 0.0723

del mod, pars, out

print("###------###")

y_min = y - np.sqrt(y)
y_max = y + np.sqrt(y)

mod = ExponentialModel()
pars = mod.guess(y_min[4:463], x=x[4:463])
out = mod.fit(y_min, pars, x=x)

print("Min: ",out.fit_report()) #amplitude = 85.2245 +/- 1.8929 , decay = 2.3260 +/- 0.0688

del mod, pars, out

mod = ExponentialModel()
pars = mod.guess(y_max[4:463], x=x[4:463])
out = mod.fit(y_max, pars, x=x)

print("Max: ",out.fit_report()) #amplitude = 102.1485 +/- 2.0711 , decay = 2.7548 +/- 0.0754

amp1 = ufloat(85.2245, 1.8929)
amp2 = ufloat(93.5841, 1.9725)
amp3 = ufloat(102.1485, 2.0711)

dec1 = ufloat(2.3260, 0.0688)
dec2 = ufloat(2.5555, 0.0723)
dec3 = ufloat(2.7548, 0.0754)

amp = (amp1 + amp2 + amp3)/3
dec = (dec1 + dec2 + dec3)/3

print("Avg Amplitude: ", "%.4f"%unumpy.nominal_values(amp), "+/-", "%.4f"%unumpy.std_devs(amp), " and Avg Decay: ", "%.4f"%unumpy.nominal_values(dec), "+/-", "%.4f"%unumpy.std_devs(dec))