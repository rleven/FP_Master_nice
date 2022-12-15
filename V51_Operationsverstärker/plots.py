import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
import uncertainties as un

##############################################################
#
# Invertierbarer Linearverstaerker
#
##############################################################

def Abweichung(x1, x2):
    return np.abs(np.abs(x1)-np.abs(x2))/x2 *100

def ampli(U1,U2): #returns amplification factor from U1 to U2
    return U2/U1 #U1 input U2 output

def fit(freq,a,b):
    return a*freq**b

def logfit(freq, a,b):
    return a*freq+b

def Grenzamplification(U1, U2):
    return U2/U1 *np.sqrt(2)

data1 = np.genfromtxt('data/inv_linear_1.csv', comments='#', delimiter=',') #freq, phase, voltage
data2 = np.genfromtxt('data/inv_linear2.csv', comments='#', delimiter=',')
data3 = np.genfromtxt('data/inv_linear3.csv', comments='#', delimiter=',')

U_in_1 = 0.141 # V for the first measurment
U_in_2 = 0.149 # V for the second measurment
U_in_3 = 0.149 # V for the third measurment

#average over plateau:
V1 = un.ufloat(np.mean(data1[:4,2]/U_in_1), np.std(data1[:4,2]/U_in_1))
V2 = un.ufloat(np.mean(data2[:8,2]/U_in_2), np.std(data2[:8,2]/U_in_2))
V3 = un.ufloat(np.mean(data3[:7,2]/U_in_3), np.std(data3[:7,2]/U_in_3))

print('Average Values of amplification: V1 ', V1, ' V2 ', V2, ' V3 ', V3)
print('calculated amplifications: V1', -100/1, ' V2 ', -100/47, ' V3 ', -220/47)
print('Abweichung exp zu theorie: V1 ', Abweichung(V1, -100), ' V2 ', Abweichung(V2,-100/47 ), ' V3 ', Abweichung(V3, -220/47) )

#fit in the decreasing area
params1, cov1 = curve_fit(fit,(data1[4:, 0]), (ampli(U_in_1, data1[4:,2])), p0=(1,1))
params2, cov2 = curve_fit(fit,(data2[8:, 0]), (ampli(U_in_2, data2[8:,2])), p0=(1,1))
params3, cov3 = curve_fit(fit,(data3[7:, 0]), (ampli(U_in_3, data3[7:,2])), p0=(1,1))

x1 = np.linspace(8000, 200000, 10000)
x2 = np.linspace(4*10**5, 1.5*10**6, 10000)
x3 = np.linspace(9*10**4, 7*10**5, 10000)

a1 = un.ufloat(params1[0], np.sqrt(cov1[0,0]))
b1 = un.ufloat(params1[1], np.sqrt(cov1[1,1]))
a2 = un.ufloat(params2[0], np.sqrt(cov2[0,0]))
b2 = un.ufloat(params2[1], np.sqrt(cov2[1,1]))
a3 = un.ufloat(params3[0], np.sqrt(cov3[0,0]))
b3 = un.ufloat(params3[1], np.sqrt(cov3[1,1]))


print('fit1 parameters: a ', un.ufloat(params1[0], np.sqrt(cov1[0,0])), ' b ', un.ufloat(params1[1], np.sqrt(cov1[1,1])))
print('fit2 parameters: a ', un.ufloat(params2[0], np.sqrt(cov2[0,0])), ' b ', un.ufloat(params2[1], np.sqrt(cov2[1,1])))
print('fit3 parameters: a ', un.ufloat(params3[0], np.sqrt(cov3[0,0])), ' b ', un.ufloat(params3[1], np.sqrt(cov3[1,1])))


# Grenzfrequenz
print('Grenzfrequenz 1 :', ((np.sqrt(100/2)/a1)**(1/b1)))
print('Grenzfrequenz 2 :', ((np.sqrt(100/2)/a2)**(1/b2)))
print('Grenzfrequenz 3 :', ((np.sqrt(100/2)/a3)**(1/b3)))

# Bandbreitenprodukt
print('Bandbreitenprodukt 1 :', V1*((np.sqrt(100/2)/a1)**(1/b1)))
print('Bandbreitenprodukt 2 :', V2*((np.sqrt(100/2)/a2)**(1/b2)))
print('Bandbreitenprodukt 3 :', V3*((np.sqrt(100/2)/a3)**(1/b3)))

# plots
data = [data1, data2, data3]
params = [params1, params2, params3]
x = ([x1, x2, x3])
for i in range(3):
    plt.figure()
    plt.plot(data[i][:,0], ampli(U_in_1, data[i][:,2]),'rx', label='Gemessene Verstärkung')
    plt.plot(x[i] , (fit(x[i], (params[i][0]), (params[i][1]))), label='Fit')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Frequenz in Hz')
    plt.ylabel('Verstärkung')
    plt.legend()
    plt.savefig('build/inv_lin_' + str(i+1) + '.pdf')
#    plt.show()
plt.close()

# Frequency depends of the phase

for i in range(3):
    plt.figure()
    plt.plot(data[i][:,0], data[i][:,1],'rx', label='Gemessene Phase')
#   plt.plot(x[i] , (fit(x[i], (params[i][0]), (params[i][1]))), label='Fit')
    plt.xscale('log')
#   plt.yscale('log')
    plt.xlabel('Frequenz in Hz')
    plt.ylabel('Phase / °')
    plt.legend()
    plt.savefig('build/inv_lin_phase' + str(i+1) + '.pdf')
#    plt.show()
plt.close()

################################################################################
#
#   Umkehr-Integrator
#
################################################################################

# Suchen Sie einen sinnvollen Frequenzbereich und machen Sie eine Ausgleichsgrade. Ermitteln
# Sie daraus die in den Vorbereitungen ermittelte Proportionalit ̈at. Zeigen Sie ihre Eingangssignale im
# Vergleich zu den resultierten Integrierten Ausgangssignale


data4 = np.genfromtxt('data/umkehr_int.csv', comments='#', delimiter=',') #u in, u out, freq

data4[:, 1] = data4[:,1]/data4[:,0]

params4, cov4 = curve_fit(fit,(data4[:, 2]), data4[:,1], p0=(1,1))
print('fit4 parameters: a ', un.ufloat(params4[0], np.sqrt(cov4[0,0])), ' b ', un.ufloat(params4[1], np.sqrt(cov4[1,1])))
x1 = np.linspace(2, 2*10**3, 10000)

plt.figure()
plt.plot(data4[:,2], data4[:,1],'rx', label='Gemessene Spannung')
plt.plot(x1, fit(x1, params4[0], params4[1]), label='Fit')
plt.xlabel('Frequenz ' + r'$f / Hz$')
plt.ylabel('Ausgangsspannung ' + r'$U_{out} / V$')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('build/inv_int.pdf')
#plt.show()

################################################################################
#
#   Umkehr-Differenzierer
#
################################################################################


data5 = np.genfromtxt('data/inv_diff.csv', comments='#', delimiter=',') #u in, u out, freq

data5[:, 1] = data5[:,1]/data5[:,0]


params5, cov5 = curve_fit(fit,(data5[:, 2]), data5[:,1], p0=(1,1))
print('fit5 parameters: a ', un.ufloat(params5[0], np.sqrt(cov5[0,0])), ' b ', un.ufloat(params5[1], np.sqrt(cov5[1,1])))
x1 = np.linspace(2, 2*10**3, 10000)

plt.figure()
plt.plot(data5[:,2], data5[:,1],'rx', label='Gemessene Spannung')
plt.plot(x1, fit(x1, params5[0], params5[1]), label='Fit')
plt.xlabel('Frequenz in Hz')
plt.ylabel('Ausgangsspannung / V')
plt.xscale('log')
plt.yscale('log')
plt.legend()
plt.savefig('build/inv_diff.pdf')
#plt.show()

plt.close()


data6 = np.genfromtxt('data/scope_16.csv', comments='#', delimiter=',', deletechars='+-') #s, u in, u out

plt.figure()
plt.plot(data6[:,0], data6[:,1],'rx', label='angelegte Spannung')
plt.plot(data6[:,0], data6[:,2],'ko', label='Output Spannung')
plt.xlabel('Zeit in s')
plt.ylabel('Spannung / V')
plt.legend()
plt.savefig('build/scope_15.pdf')
#plt.show()

######################################################################################
#
# Schmitt-Trigger
#
######################################################################################

R_1 = 10000
R_p = 100000
U_B = un.ufloat(29.1,0.05)
print('Schmitt Kippspannung: ', R_1/R_p *U_B)
print('Schmitt Abweichung ', Abweichung(3.26,  R_1/R_p *U_B))
######################################################################################
#
# Signal-Generator
#
######################################################################################

R_1 = 10000 #Ohm
R_2 = 100000
R_3 = 1000
C = 1*10**(-6) #Farad
U_max = un.ufloat(27.7,0.05)

print('Signal Generator: Frequenz ', R_2/(4*R_1*C*R_3), ' Amplitude ', U_max*(R_1/R_2))
print('Abweichung Generator: ', Abweichung(5.20, U_max*(R_1/R_2)))