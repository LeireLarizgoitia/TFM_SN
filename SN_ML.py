#!/usr/bin/env python
# coding: utf-8

# Leire Larizgoitia Arcocha

import matplotlib
matplotlib.rcParams['text.usetex'] = True
from matplotlib import pyplot as plt
#plt.rcParams.update({'font.size': 20})
import matplotlib.colors as mcolors
import matplotlib.ticker as mtick
import numpy as np
import math
import scipy.integrate as integrate
import scipy.integrate as quad
import scipy.constants as constants
import pandas as pd
import random
from scipy.stats import norm
from scipy import special
from scipy.special import gamma, factorial
import matplotlib.mlab as mlab
import statistics
from scipy.optimize import curve_fit
from scipy.special import factorial
from scipy.stats import poisson
from scipy.special import gammaln # x! = Gamma(x+1)

"Constants"
c = constants.c # speed of light in vacuum
e = constants.e #elementary charge
Na = constants.Avogadro #Avogadro's number
fs = constants.femto # 1.e-15
year = constants.year #one year in seconds
bar = constants.bar
kpc = 1e3 * constants.parsec #m
kpc_cm = kpc * 1e2 #cm
ton = constants.long_ton #one long ton in kg

h =  constants.value(u'Planck constant in eV/Hz')
hbar = h/(2*np.pi) # the Planck constant h divided by 2 pi in eV.s
hbar_c = hbar * c* 1E+9 # units MeV fm
hbar_c_ke = 6.58211899*1e-17 * c #KeV cm
Gf00 = constants.physical_constants["Fermi coupling constant"]
Gf = constants.value(u'Fermi coupling constant') # units: GeV^-2
"1GeV-2 = 0.389379mb" "10mb = 1fm^2"
Gf = Gf * 1e-12 * (hbar_c_ke)**3 #keV cm^3


"Xenon 132 isotope"
Z =  54
N =  78
A = Z+N
M =  131.9041535 * constants.u # mass of Xenon 132 in kg
M = M*c**2/e *1e-3 #mass of Xenon 132 in keV
M_u = 131.9041535 #in u

"RMS of Xenon"
Rn2 = (4.8864)**2
Rn4 = (5.2064)**4

"SUPERNOVA NEUTIRNOS"

erg_MeV = 624.15 * 1e3 # MeV

"SUPERNOVA MODEL"
alpha = 2.3
A_true = 3.9*1e11 #cm-2
Eav_true = 14 #MeV

"Energy range of the incoming neutrino in MeV (more or less)"
Enu_min = 0.0
Enu_max = 50.

"Recoil energy range of Xenon in KeV"
T_min= 1.
T_max = 1/2 * (M + 2*Enu_max*1e3 - np.sqrt(M**2 + 4*Enu_max*1e3*(M - Enu_max*1e3)))
T_thres = 1.  #THRESOLD RECOIL ENERGY FOR DETECTION

sigma0 = 0.4 #corresponds to T_thresold for Xe

"Approximation values"
sin_theta_w_square = 0.231 #zero momnetum transfer data from the paper
Qw = N - (1 - 4*sin_theta_w_square)*Z
gv_p = 1/2 - sin_theta_w_square
gv_n = -1/2

"NORMALIZATION CONSTANT"
Mass_detector = 10* 1e3 #kg
Dist = 10 #kpc
Distance = Dist * kpc_cm # Supernova at the Galactic centre in cm

print('Mass of the detector in kg : ', Mass_detector)
print('Distance to the source in kpc : ', Dist)

Area = 4*np.pi* Distance**2

Dist = Distance / kpc_cm
print('Distance to the source in kpc : ', Dist)
print('Solid angle : ', Area)
print('Solid angle reverse : ', 1/Area)

efficiency = 0.80

normalization =  Mass_detector * 1e3 * Na / (M_u * Area)

print('Normalization constant for a detector of mass ', Mass_detector, ' kg and a distance to the source of ', Dist, ' kpc : '  , normalization)

nsteps = 100

"FUNCTIONS"

def F(Q2,N,Rn2,Rn4):  # from factor of the nucleus
    Fn = N* (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2 +  Q2**2/math.factorial(5) * Rn4/hbar_c**4) #approximation
    return (Fn)

def cross_section(T,Enu, N,M,Rn2,Rn4):
    Q2= 2 *Enu**2 * M * T *1E-6 /(Enu**2 - Enu*T*1E-3) #MeV ^2
    dsigmadT = Gf**2 /(2*np.pi)  / 4 * F(Q2,N,Rn2,Rn4)**2 * M / (hbar_c_ke)**4 * (2 - 2*T *1E-3 / Enu  + (T *1E-3 / Enu)**2 - M *T * 1E-6 / Enu**2 ) #cm^2/keV
    return dsigmadT

def flux(E,A,E_av):  #Fluxes following a continuous distribution. Normalized
    NN = (alpha + 1)**(alpha + 1) /(E_av * gamma(alpha + 1))
    phi =  NN * (E/E_av)**alpha * np.exp(-(alpha + 1)*(E/E_av)) #energy spectrum
    f = A * phi
    return f

int_e =  np.zeros((nsteps+1),float)
int_antie =  np.zeros((nsteps+1),float)
int_x = np.zeros((nsteps+1),float)
EE_antie = np.zeros((nsteps+1),float)
EE_e = np.zeros((nsteps+1),float)
EE_x = np.zeros((nsteps+1),float)

def differential_events_flux(T,A, Eav,N,M,Rn2,Rn4):
    nsteps = 100
    int=  np.zeros((nsteps+1),float)
    EE = np.zeros((nsteps+1),float)
    "Integral Bounds"
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1E-3 #MeV
    Emax= Enu_max
    for i in range (0,nsteps+1):
        EE[i] = Emin + (Emax - Emin)/nsteps * i
        int[i] = (cross_section(T,EE[i],N,M,Rn2,Rn4) * flux(EE[i],A,Eav))
    return (np.trapz(int, x=EE))

"PLOT FUNCTIONS"

def closest(lst, K):
    list1=[]
    for i in range(len(lst)):
        list1.append(lst[i])
    num1 = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]
    lst.remove(num1)
    num2 = lst[min(range(len(lst)), key = lambda i: abs(lst[i]-K))]

    pos1 = -1
    pos2 = -1
    # Iterate over list items by index pos
    for i in range(len(list1)):
        if (num1 >=0.0 and num1<=5.0):
            # Check if items matches the given element
            if list1[i] == num1:
                pos1 = i
    for i in range(len(list1)):
        if (num2 >=0.0 and num2<=5.0):
            # Check if items matches the given element
            if list1[i]==num2:
                pos2 = i
    return  pos1,pos2

def Likelihood_bins(n_true,T): #bineado

    dNdT = []
    Eaverage=[]
    AT=[]
    mu=[]
    chi2=[]
    Escan=[]
    Ascan=[]
    pairse=[]
    pairsa=[]
    likelihoods =[]

    Eav_min = 5.
    Eav_max = 27 #erg_MeV
    AT_min=0.1*1e11
    AT_max = 10*1e11 #cm^-2

    nscan =100
    nsteps = 1

    for i in range(0,nscan): #range of scan
        Eaverage.append(Eav_min + (Eav_max-Eav_min)/nsteps * i)
        AT.append(AT_min + (AT_max-AT_min)/nsteps * i)

    #Scan
    for a in range(len(AT)): #all amplitudes
        for e in range(len(Eaverage)): #all average energies
            for j in range(0,len(T)-1):
                for i in range(0,nsteps+1):
                        T_bins.append(T[j] + ( T[j+1] - T[j])/nsteps * i)
                        dNdT.append(Mass_detector *1e3* Na /M_u * (differential_events_flux(T_bins[i],AT[a], Eaverage[e],N, M,Rn2,Rn4))) #normlization factor m_detector in g

                mu.append(np.trapz(dNdT, x=T_bins)) #append for each bin

                T_bins.clear()
                dNdT.clear()


            sum = 0
            sum_n = 0
            sum_tot = 0
            for i in range(0, len(T)-1):
                sum_tot = sum_tot + (mu[i] - n_true[i] + n_true[i]*np.log(n_true[i] / mu[i])) # this is lnL / lnL_max
            mu.clear()

            chi2.append(2 * sum_tot)
            likelihoods.append(2*sum_tot)

            pairse.append(Eaverage[e])
            pairsa.append(AT[a])


        #print(chi2)

        # Driver code , selection of likelihood difference to 2.3
        K = 2.3

        e1,e2 = closest(chi2, K)

        chi2.clear()

        Escan.append(Eaverage[e1])
        Escan.append(Eaverage[e2])
        Ascan.append(AT[a])
        Ascan.append(AT[a])


    plt.hist(likelihoods,label='Likelihood distribution', color='blue', alpha = 0.3)
    plt.legend()
    plt.show()

    plt.scatter(pairse, pairsa)
    plt.xlabel('ET')
    plt.ylabel('AT')
    plt.ylim(0.01*1e11,10*1e11)
    plt.xlim(5,30)
    plt.legend()
    plt.show()

    plt.scatter(Escan, Ascan)
    plt.scatter(Eav_true,A_true, color='green')
    plt.xlabel('ET')
    plt.ylabel('AT')
    plt.ylim(0.01*1e11,10*1e11)
    plt.xlim(5,30)
    plt.legend()
    plt.show()

#%%
"MAIN PART"

TT=[]
dNdT = []

nsteps = 1000
for i in range(0,nsteps+1):
    TT.append(T_thres + (T_max - T_thres)/nsteps * i)
    dNdT.append( Mass_detector *1e3* Na /M_u * (differential_events_flux(TT[i],A_true, Eav_true,N, M,Rn2,Rn4))) #normlization factor m_detector in g
total = np.trapz(dNdT, x=TT)

print('Total number of events: ', total)

"Binning"
t = 1.487929405
sigma = sigma0 * T_thres * np.sqrt(t/T_thres)
x0 = t-sigma

def x1(x0, sigma0, T_thres):
    a = 1
    b = -2*(x0+(sigma0*T_thres)**2./T_thres)
    c = (x0-2*(sigma0*T_thres)**2./T_thres)*x0
    x = (-b+np.sqrt(b*b-4*a*c))/(2*a)
    sigma = sigma0 * T_thres *np.sqrt((x0+x)/(2*T_thres))

    return [x, sigma]

bins = []
while (x0 <=T_max) :
    bin0 = x1(x0, sigma0, T_thres)
    x0 = bin0[0]
    bins.append(bin0)

binss = []
centres = []

for bint in bins:
    centres.append(bint[0]-bint[1])

bins.insert(0, [t-sigma, sigma0])

dNdT_bins=[]

for bint in bins:
    binss.append(bint[0])

for i in range(0,len(binss)-1):
    dNdT_bins.append(Mass_detector *1e3* Na /M_u * (differential_events_flux(binss[i],A_true, Eav_true,N, M,Rn2,Rn4)))


"Events on intervals"
T_bins = []
events_interval= []
dNdT = []

for j in range(0,len(binss)-1):
    for i in range(0,nsteps+1):
            T_bins.append(binss[j] + ( binss[j+1] - binss[j])/nsteps * i)
            dNdT.append(Mass_detector *1e3* Na /M_u * (differential_events_flux(T_bins[i],A_true, Eav_true,N, M,Rn2,Rn4))) #normlization factor m_detector in g

    events_interval.append(np.trapz(dNdT, x=T_bins))

    T_bins.clear()
    dNdT.clear()

print('Total number of events after binning: ', sum(events_interval))

Likelihood_bins(events_interval,binss)

"END OF CODE"
