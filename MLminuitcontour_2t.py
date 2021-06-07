#!/usr/bin/env python
# coding: utf-8

# Leire Larizgoitia Arcocha

from __future__ import division

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
from scipy import stats

from scipy.stats import norm
from scipy import special
from scipy.special import gamma, factorial
import matplotlib.mlab as mlab
import statistics
from scipy.optimize import curve_fit
from scipy.stats import poisson
from scipy.special import gammaln # x! = Gamma(x+1)
from time import time

import seaborn as sns
from statsmodels.base.model import GenericLikelihoodModel
from iminuit import Minuit

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
Rn2 = (4.8664)**2
Rn4 = (5.2064)**4
Rn6 = (5.4887)**6

"SUPERNOVA NEUTIRNOS"
erg_MeV =  624150.648 # MeV

"Energy range of the incoming neutrino in MeV (more or less)"
Enu_min = 0.0
Enu_max = 50.

"Recoil energy range of Xenon in KeV"
T_min= 1.
T_max = 1/2 * (M + 2*Enu_max*1e3 - np.sqrt(M**2 + 4*Enu_max*1e3*(M - Enu_max*1e3)))

"Approximation values"
sin_theta_w_square = 0.2386 #zero momnetum transfer data from the paper XENON1T
Qw = N - (1 - 4*sin_theta_w_square)*Z
gv_p = 1/2 - sin_theta_w_square
gv_n = -1/2

"NORMALIZATION CONSTANT"
#sigma0 = 0.4 #0.497 #actually it is simga0/T_thres
Mass_detector = 2* 1e3 #kg
Dist = 10 #kpc
Distance = Dist * kpc_cm # Supernova at the Galactic centre in cm

#print('Mass of the detector in kg : ', Mass_detector)
#print('Distance to the source in kpc : ', Dist)

Area = 4*np.pi* Distance**2

"SUPERNOVA MODEL"
#alpha = 2.3
A_true =  1 #3.9*1e11 #cm-2 irrelevant
Eav_true = 14 #MeV

Eav_truee = 11 #MeV
Eav_trueantie = 14 #MeV
Eav_truex = 15 #MeV

Ev_media = 6 / (1/Eav_truee + 1/Eav_trueantie +  4*1/Eav_truex)

#print('Weighted average of the averange energy', Ev_media)

Le = 5*1e52 *erg_MeV
Lantie = 5*1e52 *erg_MeV
Lx = 5*1e52 *erg_MeV #1.94

L = (Le + Lantie +  4*Lx)/6

At_truee = Le / Area / Eav_truee
At_trueantie = Lantie / Area / Eav_trueantie
At_truex = Lx / Area / Eav_truex

#At_media = (At_truee + At_trueantie +  4*At_truex)

At_media = 6 * L / Area / Ev_media

#print('Luminosity: ', L / erg_MeV , 'ergs')
#print('AT in cm^-2 : ', At_media)


alphae = 3
alphaantie = 3
alphax = 2
alpha_media  = 2.3 # 6 / (1/alphae + 1/alphaantie +  4*1/alphax)

#print('alpha e, antie, x : ', alphae, ',', alphaantie, ',',  alphax)

"NORMALIZATION"

Dist = Distance / kpc_cm
#print('Solid angle : ', Area)

efficiency = 0.80

normalization =  Mass_detector * 1e3 * Na / (M_u * Area)

#print('Normalization constant for a detector of mass ', Mass_detector, ' kg and a distance to the source of ', Dist, ' kpc : '  , normalization)


nsteps = 100

"FUNCTIONS"

def F(Q2,N,Rn2,Rn4):  # form factor of the nucleus
    Fn = N* (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2 +  Q2**2/math.factorial(5) * Rn4/hbar_c**4 - Q2**3/math.factorial(7) * Rn6/hbar_c**6) #approximation
    return (Fn / Qw)

def cross_section(T,Enu, N,M,Rn2,Rn4):
    Q2= 2 *Enu**2 * M * T *1E-6 /(Enu**2 - Enu*T*1E-3) #MeV ^2
    dsigmadT = Gf**2 *M /(2*4*np.pi) * Qw**2 * F(Q2,N,Rn2,Rn4)**2 / (hbar_c_ke)**4 * (2 - 2*T *1E-3 / Enu  + (T *1E-3 / Enu)**2 - M *T * 1E-6 / Enu**2 ) #cm^2/keV
    return dsigmadT

def flux_flavour(E,E_av,L,alpha):  #Fluxes following a continuous distribution. Normalized
    NL = L / E_av /Area # L already in MeV
    NN = (alpha + 1)**(alpha + 1) /(E_av * gamma(alpha + 1))
    phi =  NN * (E/E_av)**alpha * np.exp(-(alpha + 1)*(E/E_av)) #energy spectrum
    f = phi * NL
    return f

def flux_sum(E,A,E_av,alpha):  #Fluxes following a continuous distribution. Normalized
    NN = (alpha + 1)**(alpha + 1) /(E_av * gamma(alpha + 1))
    phi =  NN * (E/E_av)**alpha * np.exp(-(alpha + 1)*(E/E_av)) #energy spectrum
    f = A * phi
    return f

def differential_events_flux_flavour(T,N,M,Rn2,Rn4):
    nsteps = 20
    int=  np.zeros((nsteps+1),float)
    EE = np.zeros((nsteps+1),float)
    "Integral Bounds"
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1E-3 #MeV
    Emax= Enu_max
    for i in range (0,nsteps+1):
        EE[i] = Emin + (Emax - Emin)/nsteps * i
        int[i] = (cross_section(T,EE[i],N,M,Rn2,Rn4) * (flux_flavour(EE[i],Eav_truee,Le,alphae) + flux_flavour(EE[i],Eav_trueantie,Lantie,alphaantie) + 4*flux_flavour(EE[i],Eav_truex,Lx,alphax)))
    return (np.trapz(int, x=EE))

def differential_events_flux_sum(T,A,Eav,alpha,N,M,Rn2,Rn4):
    nsteps = 20
    int=  np.zeros((nsteps+1),float)
    EE = np.zeros((nsteps+1),float)
    "Integral Bounds"
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1E-3 #MeV
    Emax= Enu_max
    for i in range (0,nsteps+1):
        EE[i] = Emin + (Emax - Emin)/nsteps * i
        int[i] = (cross_section(T,EE[i],N,M,Rn2,Rn4) * flux_sum(EE[i],A,Eav,alpha))
    return (np.trapz(int, x=EE))

def constant_usefull():
    c = constants.c # speed of light in vacuum
    e = constants.e #elementary charge
    "RMS of Xenon"
    Rn2 = (4.8864)**2
    Rn4 = (5.2064)**4
    Z =  54
    N =  78
    Na = constants.Avogadro #Avogadro's number
    #Mass_detector =  10* 1e3 #kg
    M_u = 131.9041535 #in u
    #M =  131.9041535 * constants.u # mass of Xenon 132 in kg
    M = 131.9041535 * constants.u*c**2/e *1e-3 #mass of Xenon 132 in keV
    Enu_min = 0.0
    Enu_max = 50.
    T_max = 1/2 * (M + 2*Enu_max*1e3 - np.sqrt(M**2 + 4*Enu_max*1e3*(M - Enu_max*1e3)))

"Binning   -  Return: array of T in bins (bin size dependent)"
def binning():
    t = T_thres - 0.03
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

    for bint in bins:
        binss.append(bint[0])

    return binss,centres

def fnc_QF(FF,E): #for XENON
    k = 0.133* Z**(2/3) * A**(-1/2) # k=0.166 for Xe
    epsilon = 11.5 * E * Z**(-7/3)
    g = 3 *epsilon**(0.15) + 0.7*epsilon**(0.6) + epsilon
    L = k*g / (1 + k*g)

    alpha = 1.240
    xi = 0.0472
    beta = 239
    gamma = 0.01385
    delta = 0.0620

    Aa_up = alpha + 0.079
    Aa_lo = alpha - 0.073

    Axi_up = xi + 0.0088
    Axi_lo = xi - 0.0073

    Abeta_up = beta + 28
    Abeta_lo = beta - 8.8

    Agamma_up = gamma + 0.00058
    Agamma_lo = gamma - 0.00073

    Adelta_up = delta + 0.0056
    Adelta_lo = delta - 0.0064

    W = 13.7e-3 #keV (Dahl's thesis)

    NexNi = 0.2 #alpha * FF**(-xi) * (1 - np.exp(-beta*epsilon))


    phi = gamma * FF**(-delta)
    Ni = E * L / W   / (1 + NexNi)
    fN = 1/ (1 + NexNi)

    r = 1 - (np.log(1 + Ni * phi) / (Ni*phi))
    r_prima = np.log(1 + Ni * phi) / (Ni*phi)

    ne = L * E/W * (1/(1 + NexNi)) * r_prima #(1-r)

    Qy = ne/E

    return Qy

def fnc_L(E): #for XENON
    k = 0.133* Z**(2/3) * A**(-1/2) # k=0.166 for Xe
    epsilon = 11.5 * E * Z**(-7/3)
    g = 3 *epsilon**(0.15) + 0.7*epsilon**(0.6) + epsilon

    L = k*g / (1 + k*g)

    return L


"Events on intervals after Resolution"
def fnc_events_interval_obs():
    constant_usefull()
    binss,centres = binning()
    T_bins = []
    T_bins_plot = []
    dNdT = []
    dNdT_plot = []
    events_interval= []
    events_interval_obs= []
    dNdT_res_s = []
    dNdT_res_obs=[]

    t_obs_plot=[]
    dNdT_res_obs_plot=[]

    t_obs_plot1=[]
    dNdT_res_obs_plot1=[]

    T_true1=[]
    dNdT1 = []

    T_true=[]
    dNdT = []

    nsteps = 10000

    for i in range(0,nsteps+1):
        t = T_thres + (T_max - T_thres)/nsteps * i
        T_true.append(t)
        dNdT.append( Mass_detector *1e3* Na /M_u * (differential_events_flux_flavour(t,N, M,Rn2,Rn4)))
        x = (1 - fnc_L(t))* t
        if x >= T_thres:
            T_true1.append(x)
            dNdT1.append( Mass_detector *1e3* Na /M_u * (differential_events_flux_flavour(t,N, M,Rn2,Rn4))) #Quenching applied

    total = np.trapz(dNdT, x=T_true)
    total1 = np.trapz(dNdT1, x=T_true1)

    "Sample T -> Tobs , in the way we want"
    TT_obs=[]
    nsteps_obs=1000

    'apply QF to distribution with resolution'

    for i in range(0,nsteps_obs+1):
        t = T_true[0] + (T_true[len(T_true)-1] - T_true[0])/nsteps_obs * i
        x = (1- fnc_L(t))* t
        if x >= T_thres:
            TT_obs.append(x)

    T_bins=[]
    for e in binss:
        T_bins.append(e)

    x1 = - float('inf')
    x2 = float('inf')

    gauss_resapp=[]
    tbin=[]
    for j in range(0,len(T_bins)-1):
        for i in range(0,len(TT_obs)):
            if TT_obs[i]<=T_bins[j+1] and TT_obs[i]>=T_bins[j]:
                tbin.append(TT_obs[i])

        if len(tbin)!=0:
            for tobs in tbin: #T observed
                for i in range(0,len(T_true1)):
                    sigma = sigma0 * T_thres * np.sqrt(tobs/T_thres)

                    var2 = ((tobs - x2) / (np.sqrt(2) * sigma))
                    A2 = np.sqrt(np.pi / 2) * (-sigma) * math.erf(var2)
                    var1 = ((tobs - x1) / (np.sqrt(2) * sigma))
                    A1 = np.sqrt(np.pi / 2) * (-sigma) * math.erf(var1)
                    A = 1/ (A2 - A1)

                    gauss_res = A * np.exp(-(T_true1[i]-tobs)**2 / (2*sigma**2))
                    dNdT_res_s.append(gauss_res * dNdT1[i])

                    'proof normalization'
                    #gauss_resapp.append(gauss_res)
                #int = np.trapz(gauss_resapp, x=T_true1)
                #print('t : ', tobs , ' int: ', int)
                #gauss_resapp.clear()

                dNdT_res_obs.append(np.trapz(dNdT_res_s, x=T_true1))
                dNdT_res_s.clear()

            events_interval_obs.append(np.trapz(dNdT_res_obs, x=tbin))
        else:
            pass

        dNdT_res_obs.clear()
        tbin.clear()

    events_interval_obs_simple=[]
    for e in events_interval_obs:
        if e!=0.0:
            events_interval_obs_simple.append(e)

    return events_interval_obs_simple

def fnc_events_interval_obs_sum(A,E,alpha):
    constant_usefull()
    binss,centres = binning()
    T_bins = []
    T_bins_plot = []
    dNdT = []
    dNdT_plot = []
    events_interval= []
    events_interval_obs= []
    dNdT_res_s = []
    dNdT_res_obs=[]

    t_obs_plot=[]
    dNdT_res_obs_plot=[]

    t_obs_plot1=[]
    dNdT_res_obs_plot1=[]

    T_true1=[]
    dNdT1 = []

    T_true=[]
    dNdT = []

    nsteps = 10000

    for i in range(0,nsteps+1):
        t = T_thres + (T_max - T_thres)/nsteps * i
        T_true.append(t)
        dNdT.append( Mass_detector *1e3* Na /M_u * (differential_events_flux_sum(t,A,E,alpha,N, M,Rn2,Rn4)))
        x = (1 - fnc_L(t))* t
        if x >= T_thres:
            T_true1.append(x)
            dNdT1.append( Mass_detector *1e3* Na /M_u * (differential_events_flux_sum(t,A,E,alpha,N, M,Rn2,Rn4))) #Quenching applied

    total = np.trapz(dNdT, x=T_true)
    total1 = np.trapz(dNdT1, x=T_true1)

    "Sample T -> Tobs , in the way we want"
    TT_obs=[]
    nsteps_obs=1000

    'apply QF to distribution with resolution'

    for i in range(0,nsteps_obs+1):
        t = T_true[0] + (T_true[len(T_true)-1] - T_true[0])/nsteps_obs * i
        x = (1- fnc_L(t))* t
        if x >= T_thres:
            TT_obs.append(x)

    T_bins=[]
    for e in binss:
        T_bins.append(e)

    x1 = - float('inf')
    x2 = float('inf')

    gauss_resapp=[]
    tbin=[]
    for j in range(0,len(T_bins)-1):
        for i in range(0,len(TT_obs)):
            if TT_obs[i]<=T_bins[j+1] and TT_obs[i]>=T_bins[j]:
                tbin.append(TT_obs[i])

        if len(tbin)!=0:
            for tobs in tbin: #T observed
                for i in range(0,len(T_true1)):
                    sigma = sigma0 * T_thres * np.sqrt(tobs/T_thres)

                    var2 = ((tobs - x2) / (np.sqrt(2) * sigma))
                    A2 = np.sqrt(np.pi / 2) * (-sigma) * math.erf(var2)
                    var1 = ((tobs - x1) / (np.sqrt(2) * sigma))
                    A1 = np.sqrt(np.pi / 2) * (-sigma) * math.erf(var1)
                    A = 1/ (A2 - A1)

                    gauss_res = A * np.exp(-(T_true1[i]-tobs)**2 / (2*sigma**2))
                    dNdT_res_s.append(gauss_res * dNdT1[i])


                dNdT_res_obs.append(np.trapz(dNdT_res_s, x=T_true1))
                dNdT_res_s.clear()

            events_interval_obs.append(np.trapz(dNdT_res_obs, x=tbin))
        else:
            pass

        dNdT_res_obs.clear()
        tbin.clear()

    events_interval_obs_simple=[]
    for e in events_interval_obs:
        if e!=0.0:
            events_interval_obs_simple.append(e)

    return events_interval_obs_simple

#%%
"MAIN PART"

"Finding Threshold in terms of electron number"
Tnr=[]
npoint =10000

for i in range(0,npoint+1):
    Tnr.append(0.001 + (10 - 0.001)/npoint * i)

W = 13.7e-3 #keV

Field=1000 #V/cm

ne=[]
intne=[]
for i in range(0,len(Tnr)):
    ne.append(fnc_QF(Field,Tnr[i]) * Tnr[i])

number_elec = 7.0 #number of electrons we want to be produced in IONIZATION

neup = number_elec + 0.01 #upper
nelow = number_elec - 0.01 #lower

nee=[]
for i in range(0,len(ne)):
    if ne[i]<=neup and ne[i]>=nelow:
        nee.append(ne[i])
nee.sort()

ne_thres = nee[0]

for i in range(0,len(ne)):
    if ne[i]==ne_thres:
        Tnr_thres = Tnr[i]
        break

FF = Field #V/cm ELECTRIC FIELD
T_thres = Tnr_thres # T THRESOLD NOW!

sigma1 = 0.3229230237677073 #sigma/T at 1keV
sigma0 = sigma1*np.sqrt(1/T_thres) #sigma/T at threshold

"OBSERVED EVENTS"

#print('nobs: ',sum(fnc_events_interval_obs()))

"MINUIT"

def fcn_np(par):
    constant_usefull()
    at=par[0]
    ev=par[1]
    n_obs = []
    mu = []

    n_obs = fnc_events_interval_obs()
    mu = fnc_events_interval_obs_sum(at,ev,alpha_media) #events_est
    sum_tot=0
    for i in range(0,len(n_obs)): #sum over bins
        sum_tot = sum_tot + (mu[i] - n_obs[i] + n_obs[i]*np.log(n_obs[i] / mu[i])) # this is lnL / lnL_max
        chi2 = 2*sum_tot
    return chi2

fcn_np.errordef = 1 #Minuit.LIKELIHOOD

at_start = 3.8*1e11
ev_start = 15

m = Minuit(fcn_np, (at_start,ev_start),name=("a", "b")) #

m.limits['a'] = (1, None)
m.limits['b'] = (1, None)

m.migrad()  # run optimiser
#m.simplex().migrad()  # run optimiser
#print(m.values)

a_ML = m.values[0] #ESTIMATED PARAMETERS
e_ML = m.values[1]

#m.hesse()   # assumes gaussian distribution, not adecuate, ours POISSON
m.minos()   # run covariance estimator
#print(m.errors)

a_err = m.errors[0]
e_err = m.errors[1]

"Save contour data"

vlist=[]
elist=[]

vlist.append(At_media)
elist.append(Ev_media)
vlist.append(a_ML)
elist.append(e_ML)

cv=[]
ce=[]

for i in range(0,len(vlist)):
    cv.append(vlist[i])
    ce.append(elist[i])

c=[cv,ce]

with open('WARM_Mass'+str(int(Mass_detector*1e-3))+'t_'+str(Field)+'Vcm_L'+str(ne_thres)+'electrons_res'+str(sigma0)+'_ML.txt', "w") as file:
    for x in zip(*c):
        file.write("{0} {1}\n".format(*x))

file.close()


"CONTOUR PLOT"
grid1 = m.mncontour('a','b', cl=0.6827)  #1SIGMA

"Save contour data"

with open('WARM_Mass'+str(int(Mass_detector*1e-3))+'t_'+str(Field)+'Vcm_L'+str(ne_thres)+'electrons_res'+str(sigma0)+'_contour.txt', "w") as txt_file:
    for line in grid1:
        content = str(line)
        txt_file.write(" ".join(content) + "\n") #AT, Eav

txt_file.close()


"END OF CODE"
