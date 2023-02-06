import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties as un
from uncertainties import unumpy
from uncertainties import ufloat 
#############################################################

# Kotrast berechnung

#############################################################

data = np.genfromtxt('data/kontrast_polfilter.txt', comments='#')

print(np.shape(data))

def Kontrast(I_min,I_max):
    return np.abs(I_min-I_max)/(I_min+I_max)

#makiere das Maximum des Kontrasts
maxi = np.argwhere(Kontrast(data[:,1], data[:,2]) == np.max(Kontrast(data[:,1], data[:,2])))
print(maxi[0])
print(np.max(Kontrast(data[:,1], data[:,2])))

phi_rad = np.radians(data[:,0])

def Delta_phi(phi,A):
    return A*np.abs(np.cos(phi)*np.sin(phi))


Kontrast1 = Kontrast(data[:,1], data[:,2])
params, cov = curve_fit(Delta_phi,phi_rad,Kontrast1)
phi_rad_lin = np.linspace(0,np.pi,1000)

A_fit = un.ufloat(params,np.absolute(cov)**0.5)
print('Fit Kontrast: ', A_fit)


plt.figure()
plt.plot(data[:,0], Kontrast(data[:,1], data[:,2]),'k.', label='Kontrast')
plt.plot(data[maxi[0],0], np.max(Kontrast(data[:,1], data[:,2])),'rx', label='Maximum des Kontrasts')
plt.plot(np.rad2deg(phi_rad_lin),Delta_phi(phi_rad_lin,A_fit.n), label="Fit der Messwerte")
plt.legend()
plt.xlabel('Winkel des Polfilters / °')
plt.ylabel('Kontrast')
plt.grid()
plt.tight_layout()
plt.savefig('build/Kontrast.pdf')
plt.close()

phi_rad = np.radians(data[:,0])

def Delta_phi(phi,A):
    return A*np.abs(np.cos(phi)*np.sin(phi))

Kontrast = Kontrast(data[:,1], data[:,2])

params, cov = curve_fit(Delta_phi,phi_rad,Kontrast)
phi_rad_lin = np.linspace(0,np.pi,1000)

A_fit = un.ufloat(params,np.absolute(cov)**0.5)
print(A_fit)
plt.plot(phi_rad,Kontrast,'x',label="Messwerte")
plt.plot(phi_rad_lin,Delta_phi(phi_rad_lin,A_fit.n), label="Ausgleichsrechnung")
plt.legend()
plt.tight_layout()
plt.savefig("build/Kontrast_ausgleichsrechung.pdf")
plt.close()

##############################################################

# Brechungsindex Glas

##############################################################


counts_glas = np.genfromtxt('data/glas.txt')

lambda_0 = 632.99 * 10**(-9) #m 
theta_0 = np.radians(10) #rad
D = 1 * 10**(-3) #m

def n_func(counts,theta):
    return 1/(1-(counts*lambda_0)/(2*D*theta*theta_0))

n_glas = n_func(counts_glas,theta_0)

print(n_glas)

n_glas_mean = np.mean(n_glas)
n_glas_err = np.std(n_glas)
n_glas_exp=np.array([n_glas_mean,n_glas_err])
n_glas_u = ufloat(n_glas_mean,n_glas_err)

print('Brechungsindex Glas: ', n_glas_u,'\n\n\n')

###############################################################

# Brechungsindex Luft

##############################################################

L = un.ufloat(0.1,0.0001) #m
T = 19.2+273.15 #K
p, counts1, counts2, counts3 = np.genfromtxt("data/mess_gas.txt", delimiter='\t', unpack=True)
lambda_0 = 632.99 * 10**(-9) #m 
p = p*10**(2) #pascal
def n_func(counts):
    return (counts*lambda_0)/L+1

def lorentz(p, A, T=19.2+273.15):
    R = 8.31446261815324 #SI
    #print('T: ',T)
    return (1 + 3*A*p/(R*T))**(1/2)



counts = [counts1, counts2, counts3]

plt.figure()
A_fit_mean = unumpy.uarray(np.zeros(3), np.zeros(3))
p_lin = np.linspace(np.min(p), np.max(p), 10000)
n_mess = unumpy.uarray(np.zeros(3), np.zeros(3))
for i in range(3):
    n_luft1 = n_func(counts[i])
    print('Brechungsindex Luft Bei Messung ', str(i+1) ,n_luft1)
    n_luft1_n = unumpy.nominal_values(n_luft1)

    params, cov = curve_fit(lorentz,p,np.array(n_luft1_n), T)
    A_fit = un.ufloat(params,np.absolute(cov)**0.5)
    print('A_fit : ',A_fit)
    n_mess[i] = lorentz(100000, A_fit, T)
    print('Brechungsindex bei Normaldruck für Messung ' + str(i+1) +': ', lorentz(100000, A_fit, T))
    plt.plot(p,n_luft1_n, 'bx', label = 'n der Messung ' + str(i+1))
    plt.plot(p_lin,lorentz(p_lin, unumpy.nominal_values(A_fit)), label = 'Fit, Messung ' + str(i+1))
    A_fit_mean[i] =(A_fit)
    plt.legend()
    plt.xlabel('Druck / ' + r'$\mathrm{Pa}$')
    plt.ylabel('Brechungsindex n')

plt.savefig("build/Gas.pdf")

A = np.mean(A_fit_mean)
print(A)
normaldruck = lorentz(101325, A, T=15+273.15) #normalatmosspähre !

print('Brechungsindex bei Normaldruck' ,normaldruck)

print('Mittelwert der drei Messungen: ', np.mean(n_mess) )


def abw(n_2, n_1):
    return (n_1-n_2)/(n_1+n_2)

print(abw(1.000292,np.mean(normaldruck)))