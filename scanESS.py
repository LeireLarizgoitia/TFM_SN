#!/usr/bin/env python
# coding: utf-8

# Leire Larizgoitia Arcocha

"""ANALYSIS FOR 132 Xe ISOTOPE"""

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
import matplotlib.mlab as mlab

from iminuit import Minuit, describe

"Constants"
c = constants.c # speed of light in vacuum
e = constants.e #elementary charge
fs = constants.femto # 1.e-15
year = constants.year #one year in seconds
bar = constants.bar

h =  constants.value(u'Planck constant in eV/Hz')
hbar = h/(2*np.pi) # the Planck constant h divided by 2 pi in eV.s
hbar_c = hbar * c* 1E+9 # units MeV fm
hbar_c_ke = 6.58211899*1e-17 * c #KeV cm
Gf00 = constants.physical_constants["Fermi coupling constant"]
Gf = constants.value(u'Fermi coupling constant') # units: GeV^-2
#1GeV-2 = 0.389379mb
#10mb = 1fm^2
Gf = Gf * 1e-12 * (hbar_c_ke)**3

"Masses"
m_pion = 139.57018 #MeV/c^2
m_muon = 105.6583755 #MeV/c^2

"Xenon 132 isotope"
Z= 54
N = 78
A = Z + N
M = 131.9041535 * constants.u # mass of Xenon 132 in kg
M = M*c**2/e *1e-3 #mass of Xenon 132 in keV

"QF"
QF = 0.2 #QF in Xe , 20%

"RMS of Xenon"
Rn2 = (4.8864)**2
Rn4 = (5.2064)**4

"Energy range of the incoming neutrino in MeV (more or less)"
Enu_min = 0.0
Enu_max = m_muon/2 #~52.8MeV
Enu_mu = (m_pion**2 - m_muon**2) / (2*m_pion) #~29.8MeV

"Recoil energy range of Xenon in KeV"
T_max = 1/2 * (M + 2*Enu_max*1e3 - np.sqrt(M**2 + 4*Enu_max*1e3*(M - Enu_max*1e3)))
T_thres = 0.9 #THRESOLD RECOIL ENERGY FOR DETECTION
Eee_thres = T_thres * QF

sigma0 = 0.4 #corresponds to T_thresold for Xe

"Approximation values"
sin_theta_w_square = 0.23867 #zero momnetum transfer data from the paper
Qw = N - (1 - 4*sin_theta_w_square)*Z
gv_p = 1/2 - 2*sin_theta_w_square
gv_n = -1/2

"NEUTRINOS PER YEAR IN 3 YEARS"
"ESS operateS on a scheduled 5000 hours of beam delivery per year "
Power = 5e6 #Watt
E_proton = 2e9 * e # Joule
nu_per_flavour_per_proton = 0.3
year = 3
in_operation = 5000 * 3600 * year

nu_per_flavour_per_year  =  nu_per_flavour_per_proton * Power/E_proton * in_operation

"NORMALIZATION CONSTANT"
Mass_detector = 20 #kg
Pressure = 20 #bar
Distance = 2000 #cm
Area =  75*50 #cm^2

Solid_angle =  Area / (4*np.pi* Distance**2)

nucleus = Mass_detector/(M *e/c**2*1e3)
nucleus_per_area = nucleus / Area
Nu_on_target = nucleus_per_area * Solid_angle

efficiency = 0.80

normalization = Nu_on_target * nu_per_flavour_per_year * efficiency

#print('Normalization constant for a detector of mass ', Mass_detector, ' kg and a distance to the source of ', Distance, ' m : '  , normalization)

nsteps = 100

"FUNCTIONS"

def FF(Q2):  # from factor of the nucleus
    Fn = N * (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2 +  Q2**2/math.factorial(5) * Rn4/hbar_c**4) #approximation
    return (Fn)

def F2(Q2): # second order approximation
    Fn = N * (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2) #approximation
    return (Fn)

def F(Q2,AtomicNumb):  # form factor of the nucleus
    #Fn = N* (1 - Q2/math.factorial(3) * Rn2 /hbar_c**2 +  Q2**2/math.factorial(5) * Rn4/hbar_c**4 ) #- Q2**3/math.factorial(7) * Rn6/hbar_c**6) #approximation
    q = np.sqrt(Q2)
    aa = 0.7 #fm
    r0 = 1.3 #fm
    rho = 3 / (4*np.pi *r0**3)
    R = AtomicNumb**(1/3)*r0
    #    print("A = ", AtomicNumb, "R = ", R)
    FF = 4*np.pi * rho / AtomicNumb / q**3 *hbar_c**3 * (math.sin(q*R/hbar_c) - q*R/hbar_c*math.cos(q*R/hbar_c)) * 1/ (1 + aa**2 *q**2/hbar_c**2)

    return (FF )

def cross_section_SM(T,Enu):
    Q2= 2 *Enu**2 * M * T *1E-6 /(Enu**2 - Enu*T*1E-3) #MeV ^2
    dsigmadT = Gf**2 /(2*np.pi) /Qw**2 / 4 * FF(Q2)**2 * M / (hbar_c_ke)**4 * (2 - 2*T *1E-3 / Enu  + (T *1E-3 / Enu)**2 - M *T * 1E-6 / Enu**2 ) #cm^2/keV
    return dsigmadT

def cross_section(T,Enu):
    Q2= 2 *Enu**2 * M * T *1E-6 /(Enu**2 - Enu*T*1E-3) #MeV ^2
    dsigmadT = Gf**2 *M /(4*np.pi) *(F(Q2,A))**2 / (hbar_c_ke)**4 * (1 -  M*T*1E-6 / (2*Enu**2) - T*1e-3/Enu + T**2/(2*Enu**2*1e6)) #cm^2/keV

    return dsigmadT

def flux(E,alpha):  #Fluxes following a continuous distribution. Normalized
    if (alpha == 1): #muon - antineutrino
        f = 64 / m_muon * ((E / m_muon)**2 * (3/4 - E/m_muon))
    if (alpha == 2): #electron neutrino
        f = 192 / m_muon * ((E / m_muon)**2 * (1/2 - E/m_muon))
    return f


int_mu =  np.zeros((nsteps+1),float)
int_antimu =  np.zeros((nsteps+1),float)
int_e = np.zeros((nsteps+1),float)
EE_antimu= np.zeros((nsteps+1),float)
EE_e= np.zeros((nsteps+1),float)

def differential_events(T,a):
    """Integral Bounds"""
    Emin = 1/2 * (T + np.sqrt(T**2 + 2*T*M)) * 1E-3 #MeV
    Tnu_mu = 1/2 * (M + 2*Enu_mu*1e3 - np.sqrt(M**2 + 4*Enu_mu*1e3*(M - Enu_mu*1e3)))

    if (a==0):
        if (T<Tnu_mu):
            return (cross_section(T,Enu_mu)) #~29.8MeV
        else:
            return 0.0
    if (a==1):
        for i in range (0,nsteps+1):
            EE_antimu[i] = Emin + (Enu_max - Emin)/nsteps * i
            int_antimu[i] = (cross_section(T,EE_antimu[i]) * flux(EE_antimu[i],1))
        return (np.trapz(int_antimu, x=EE_antimu))
    if (a==2):
        for i in range (0,nsteps+1):
            EE_e[i] = Emin + (Enu_max - Emin)/nsteps * i
            int_e[i] = (cross_section(T,EE_e[i]) * flux(EE_e[i],2))
        return (np.trapz(int_e, x=EE_e))


#%%
"MAIN PART"

def binningNR():
    t = 1.339129405
    sigma = sigma0 * T_thres * np.sqrt(t/T_thres)
    x0 = t-sigma

    #print('sigma', sigma)

    def x1(x0, sigma0, T_thres):
        a = 1
        b = -2*(x0+(sigma0*T_thres)**2./T_thres)
        c = (x0-2*(sigma0*T_thres)**2./T_thres)*x0
        x = (-b+np.sqrt(b*b-4*a*c))/(2*a)
        sigma = sigma0 * T_thres *np.sqrt((x0+x)/(2*T_thres))

        return [x, sigma]

    bins = []
    while (x0 < T_max) :
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

    return binss, centres


def binning():
    binss , centres = binningNR()
    Eee_thres = T_thres * QF
    centres_copy =[]
    centres_copy.clear()
    binss_copy =[]
    binss_copy.clear()
    for t in centres:
        centres_copy.append(t)
    for t in binss:
        binss_copy.append(t)
    centres = []
    centres.clear()
    for i in range(len(centres_copy)):
        centres_new = QF * centres_copy[i]
        centres.append(centres_new)
    binss = []
    binss.clear()
    binss.append(Eee_thres) #starts at T_thresold shifted
    for t in centres:
        sigma = sigma0 * Eee_thres * np.sqrt(t/Eee_thres)
        binss.append(t + sigma)

    return binss, centres


def fnc_events(epsilon_mu=0.0,epsilon_e=0.0):
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

    nsteps = 100

    Eee_thres = T_thres * QF
    Eee_max = T_max * QF


    binsNR, centreNR = binningNR()
    binss , centre = binning()

    "Sample T -> Tobs , in the way we want"
    tbin=[]
    nsteps_obs=1000

    x1 = - float('inf')
    x2 = float('inf')

    events_interval_mu = []
    events_interval_antimu = []
    events_interval_e = []
    events_interval_tot = []


    for j in range(0,len(binss)-1):
        tbin = []
        for i in range(0,nsteps_obs+1):
            tbin.append(binss[j] + (binss[j+1] - binss[j])/nsteps_obs * i)

        T_bins = []
        Eee_bins = []
        dNdT_mu = []
        dNdT_e = []
        dNdT_antimu = []
        dNdT = []

        gvn = - 1/2
        gvp = 1/2 - 2*sin_theta_w_square

        Qw2_mu = 4 * ( Z*(gvp + 2*epsilon_mu) + N*(gvn + epsilon_mu))**2
        Qw2_e = 4 * ( Z*(gvp + 2*epsilon_e) + N*(gvn + epsilon_e))**2

        for i in range(0,nsteps+1):
            T_bins.append(binsNR[j] + (binsNR[j+1] - binsNR[j])/nsteps * i)
            Eee_bins.append(binss[j] + (binss[j+1] - binss[j])/nsteps * i)

            dNdT_mu.append( normalization * (differential_events(T_bins[i],0)) * Qw2_mu )
            dNdT_antimu.append( normalization * (differential_events(T_bins[i],1)) * Qw2_mu )
            dNdT_e.append( normalization * (differential_events(T_bins[i],2)) * Qw2_e )

            dNdT.append((dNdT_mu[i] + dNdT_antimu[i] + dNdT_e[i] ) * (1 / QF))

        for tobs in tbin: #T observed
            for i in range(0,len(Eee_bins)):
                sigma = sigma0 * Eee_thres * np.sqrt(Eee_bins[i]/Eee_thres)

                var2 = ((Eee_bins[i] - x2) / (np.sqrt(2) * sigma))
                A2 = np.sqrt(np.pi / 2) * (-sigma) * math.erf(var2)
                var1 = ((Eee_bins[i] - x1) / (np.sqrt(2) * sigma))
                A1 = np.sqrt(np.pi / 2) * (-sigma) * math.erf(var1)
                A = 1/ (A2 - A1)

                gauss_res = A * np.exp(-(Eee_bins[i]-tobs)**2 / (2*sigma**2))
                dNdT_res_s.append(gauss_res * dNdT[i])

            dNdT_res_obs.append(np.trapz(dNdT_res_s, x=Eee_bins))
            dNdT_res_s.clear()

        events_interval_obs.append(np.trapz(dNdT_res_obs, x=tbin))

        dNdT_res_obs.clear()
        tbin.clear()

        events_interval_tot.append(np.trapz(dNdT, x=Eee_bins))

    return events_interval_obs


binss , centres = binning()
n_obs = fnc_events()

def fnc_events_estimate(epsilon_mu,epsilon_e):
    events_total_est = fnc_events(epsilon_mu,epsilon_e)
    return events_total_est

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
        if (num1 >=2.27 and num1<=2.33):
            # Check if items matches the given element
            if list1[i] == num1:
                pos1 = i
            else:
                pass
        else:
            pass
    for i in range(len(list1)):
        if (num2 >=2.27 and num2<=2.33):
            # Check if items matches the given element
            if list1[i]==num2:
                pos2 = i
            else:
                pass
        else:
            pass

    return  pos1,pos2


def Likelihood_bins(n_true): #bineado

    eps_ee=[]
    eps_mumu=[]
    mu=[]
    chi2=[]
    Escan=[]
    Ascan=[]
    pairse=[]
    pairsa=[]
    likelihoods =[]

    ch=[]

    eps_ee_min = -0.4
    eps_ee_max = 0.6
    eps_mumu_min = -0.1
    eps_mumu_max = 0.5

    nscan =100

    for i in range(0,nscan): #range of scan
        eps_ee.append(eps_ee_min + (eps_ee_max - eps_ee_min)/nscan * i)
        eps_mumu.append(eps_mumu_min + (eps_mumu_max -eps_mumu_min)/nscan * i)

    #Scan
    for e_mumu in (eps_mumu): #all
        for e_ee in (eps_ee): #all
            mu = fnc_events(e_mumu,e_ee)

            sum_tot = 0
            for i in range(0, len(n_true)):
                sum_tot = sum_tot + (mu[i] - n_true[i] + n_true[i]*np.log(n_true[i] / mu[i])) # this is lnL / lnL_max
            mu.clear()

            chi2.append(2 * sum_tot)
            likelihoods.append(2*sum_tot)
            pairse.append(e_ee)
            pairsa.append(e_mumu)

        # Driver code , selection of likelihood difference to 2.3
        K = 2.3

        e1,e2 = closest(chi2, K)

        #print(chi2)

        chi2.clear()

        ch.append(e1)
        ch.append(e2)

        if (e1!=-1):
            Escan.append(eps_ee[e1])
            Ascan.append(e_mumu)
        if (e2!=-1):
            Escan.append(eps_ee[e2])
            Ascan.append(e_mumu)
        else:
            pass

    T_obs=[]
    dNdT_est=[]
    dNdT_obs=[]
    n_obs = []
    mu=[]
    chi2=[]

    for aa in range(0,len(Ascan)):
        a_obs = Ascan[aa]
        e_obs = Escan[aa]

        mu = fnc_events(a_obs,e_obs)

        sum_tot=0
        for ii in range(0, len(n_true)): #sum over bins
            sum_tot = sum_tot + (mu[ii] - n_true[ii] + n_true[ii]*np.log(n_true[ii] / mu[ii])) # this is lnL / lnL_max
        chi2.append(2*sum_tot)
        mu.clear()
        #return chi2
    #print(chi2)

    txt_file = open("ESScontour_scan.txt", "w")

    for aa in range(0,len(Ascan)):
        a_obs = Ascan[aa]
        e_obs = Escan[aa]

        scan = [a_obs,e_obs]
        content = str(scan)
        txt_file.write(" ".join(content) + "\n")

    txt_file.close()

Likelihood_bins(n_obs)
