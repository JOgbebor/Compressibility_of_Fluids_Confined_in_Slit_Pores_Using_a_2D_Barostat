# -*- coding: utf-8 -*-
"""
Created on Thu Jun 22 21:36:26 2023

@author: jeogb
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy.constants as c
from uncertainties import ufloat
import csv
from CoolProp.CoolProp import PropsSI
import statsmodels.api as sm
from uncertainties.umath import *
from uncertainties import unumpy

if True: # Plotting style guide

    style = {
        "figure.figsize": (10, 8),
        "font.size": 36,
        "axes.labelsize": "26",
        "axes.titlesize": "26",
        "xtick.labelsize": "26",
        "ytick.labelsize": "26",
        "legend.fontsize": "18",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "lines.linewidth": 2,
        "lines.markersize": 12,
        "xtick.major.size": 18,
        "ytick.major.size": 18,
        "xtick.top": True,
        "ytick.right": True,
    }
    
    matplotlib.rcParams.update(style)
    markers = ["o", "X",  "^", "P", "d", "*", "s", ".", "x", ">"] * 5
    matplotlib.rcParams["font.family"] = ["serif"]

#============================================================================#

H = 2 # nm

if True:
    
    sig_ff = 3.40 # Ang
    sig_sf = 3.40 # Ang
    
    Shape = "Cylinder"
    
    T = 119.6 # K
    
    N = [1,2,3,4,5,6,7,8,9,10]
    
    I = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    P_sat = 1186438.11 # Pa
    
    P = [i*P_sat for i in I] # Pa
    
    offset = 10e6
    
    if Shape == "Slit":
        
        N = [1,2,3,4,5,6,7,8,9]
        
        P = [131835.345, 263660.691, 395486.036, 527311.381, 659136.726, 790962.072, 922787.417, 1054612.76, 1186438.11]
        
        offset = 1e6
    
    mean_V = []
    std_V = []
    
    K = []
    K_err = []
    Rel_Err = []
    
    for n,p,i in zip(N,P,I):
        
        pressure = round(p/1e6, 3)
        
        file = f"//wsl$/Ubuntu-20.04/home/jeo27/Confined_Argon_{Shape}_Backup_7-6-2023/{H}nm/log_{n}.lammps"
        
        if Shape == "Slit":
            Data = np.loadtxt(file, skiprows=6, max_rows=int(1e7/100))
            if H == 2:
                H_eff = 18.2502
                V = [(v * H_eff / (H*10)) for v in Data[:,2]] # Ang^3
            if H == 3:
                H_eff = 28.2498
                V = [(v * H_eff / (H*10)) for v in Data[:,2]] # Ang^3
            if H == 4:
                H_eff = 38.2498
                V = [(v * H_eff / (H*10)) for v in Data[:,2]] # Ang^3
            if H == 5:
                H_eff = 48.2498
                V = [(v * H_eff / (H*10)) for v in Data[:,2]] # Ang^3
                
        if Shape == "Cylinder":
            Data = np.loadtxt(file, skiprows=8, max_rows=int(2e7/100))
            V = [((v / (H*10)**2) * c.pi * ((H*10 - 1.7168*sig_sf + sig_ff)/2)**2) for v in Data[:,2]] # Ang^3
        
        Time = [(t-offset)*1e-6 for t in Data[:,0]]
        
        fig = plt.figure()
        plt.plot(Time, V, color="blue")
        plt.xlabel("Time (ns)")
        plt.ylabel(r"V ($\AA^3$)")
        plt.title(f"{H} nm, p/p0 = {i}", loc="right")
        plt.savefig(f"C:/Confined_Argon_Data/Cylinders_Backup_7-5-2023/{H}nm/V_{H}nm_{n}.pdf", bbox_inches="tight")
        plt.show(fig)
        plt.close(fig)
        
        fig = plt.figure()
        plt.hist([v for v,t in zip(V,Time) if 5 < t < 19], color="blue", bins=50, density="True")
        plt.xlabel(r"V ($\AA^3$)")
        plt.ylabel("Probability Density")
        plt.title(f"{H} nm, p/p0 = {i}", loc="right")
        plt.savefig(f"C:/Confined_Argon_Data/Cylinders_Backup_7-5-2023/{H}nm/Vhist_{H}nm_{n}.pdf", bbox_inches="tight")
        plt.show(fig)
        plt.close(fig)
        
        mean_v = np.mean([v for v,t in zip(V,Time) if 5 < t < 19])
        std_v = np.std([v for v,t in zip(V,Time) if 5 < t < 19])
        ufloat_std = sqrt(np.sum([abs(v-ufloat(mean_v,std_v))**2 for v,t in zip(V,Time) if 5 < t < 19])/(len([v for v,t in zip(V,Time) if 5 < t < 19])-1))
        
        k = ((c.k * T * ufloat(mean_v,std_v) * 1e-30) / ((ufloat_std * 1e-30)**2)) * 1e-9 # GPa
        
        mean_V.append(mean_v)
        std_V.append(std_v)
        K.append(k)
        K_err.append(k.s)
        
        lags = 80
        c_i = sm.tsa.acf(V, nlags=lags, fft=True)
        tau = (1/2) + np.sum(c_i[1:])
        rel_err = np.sqrt(2*tau/len(V))
        
        Rel_Err.append(rel_err)
    
#============================================================================#

if True:

    file_name = f"C:/Confined_Argon_Data/Cylinders_Backup_7-5-2023/{H}nm/{H}nm_Data.csv"

    with open(file_name, "w", newline="") as csvfile:
    
        csvwriter = csv.writer(csvfile)
    
        Header2 = ["Pressure (Pa)", "Bulk Modulus (GPa)", "Uncertainty (GPa)", "Mean Volume (A^3)", "Std Volume (A^3)"]
        csvwriter.writerow(Header2)
    
        for p,k,k_std,mean_v,std_v in zip(P, [k.nominal_value for k in K], [k.s for k in K], mean_V, std_V):
    
            Row = [p, k, k_std, mean_v, std_v]
            csvwriter.writerow(Row)

#============================================================================#

if Shape == "Cylinder":
    
    file = f"C:/Confined_Argon_Data/Cylinders/{H}nm/{H}nm_Data.csv"
    Data = np.loadtxt(file, delimiter=",", skiprows=1)
    
    file_Dobr = f"C:/Confined_Argon_Data/Cylinders/{H}nm/{H}nm_Dobr_Data.csv"
    Data_Dobr = np.loadtxt(file_Dobr, delimiter=",", skiprows=1)
    
    fig = plt.figure()
    plt.errorbar([p/max(Data[:,0]) for p in Data[:,0]], Data[:,1], yerr=Data[:,2], capsize=7.0, color="blue", linestyle="--", marker="o", label="This work")
    plt.plot(Data_Dobr[:,0], Data_Dobr[:,1], color="red", linestyle="--", marker="o", label="Dobrzanski 2018")
    plt.xlabel("p/p$_0$")
    plt.ylabel(r"K$_{\rm T}$ (GPa)")
    plt.title(f"{H} nm, T = 119.6 K", loc="right")
    plt.legend(loc="best")
    #plt.savefig(f"C:/Confined_Argon_Data/{Shape}s/{H}nm/KvP_{H}nm.pdf", bbox_inches="tight")
    plt.show(fig)
    plt.close(fig)

#============================================================================#

if True:
    
    fig = plt.figure()
    plt.plot([p/max(P) for p in P], [k for k in K_err], color="blue", linestyle="--", marker="o", label="Propagation")
    plt.plot([p/max(P) for p in P], [k.nominal_value*e for k,e in zip(K,Rel_Err)], color="red", linestyle="--", marker="o", label="AC")
    plt.xlabel("p/p$_0$")
    plt.ylabel(r"Error, $\delta$ K (GPa)")
    plt.title(f"{H} nm, T = 119.6 K", loc="right")
    plt.legend(loc="best")
    plt.savefig(f"C:/Confined_Argon_Data/{Shape}s/{H}nm/{H}nm_Error_Comparison.pdf", bbox_inches="tight")
    plt.show(fig)
    plt.close(fig)









