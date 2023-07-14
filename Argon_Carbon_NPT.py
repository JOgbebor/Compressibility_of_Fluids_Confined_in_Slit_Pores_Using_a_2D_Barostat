# -*- coding: utf-8 -*-
"""
Created on Sat Jun 24 10:44:50 2023

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

if True:

    H = 2 # nm
    
    T = 119.6 # K
    
    N = [1,2,3,4,5,6,7,8,9,10]
    
    P = [0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0]
    
    P_sat = 11.71
    
    P = [round(p*P_sat, 3) for p in P]
    
    mean_V = []
    std_V = []
    
    K = []
    
    for n,p in zip(N,P):
        
        if n < 10:
            file = f"//wsl$/Ubuntu-20.04/home/jeo27/Argon_Carbon_NPT/{H}nm/log_0{n}.lammps"
            
        if n == 10:
            file = f"//wsl$/Ubuntu-20.04/home/jeo27/Argon_Carbon_NPT/{H}nm/log_1.lammps"
    
        Data = np.loadtxt(file, skiprows=4, max_rows=int(1e7/100))
        
        Time = [(t-5e5)*1e-6 for t in Data[:,0]]
        V = [v for v in Data[:,2]]
        
        fig = plt.figure()
        plt.plot(Time, V, color="blue")
        plt.xlabel("Time (ns)")
        plt.ylabel(r"V ($\AA^3$)")
        plt.title(f"{H} nm, P = {p} atm", loc="right")
        plt.savefig(f"C:/Argon_Carbon/{H}nm/V_{H}nm_{n}.pdf", bbox_inches="tight")
        plt.show(fig)
        plt.close(fig)
        
        fig = plt.figure()
        plt.hist(V, color="blue", bins=50, density="True")
        plt.xlabel(r"V ($\AA^3$)")
        plt.ylabel("Probability Density")
        plt.title(f"{H} nm, P = {p} atm", loc="right")
        plt.savefig(f"C:/Argon_Carbon/{H}nm/Vhist_{H}nm_{n}.pdf", bbox_inches="tight")
        plt.show(fig)
        plt.close(fig)
        
        mean_v = np.mean(V)
        std_v = np.std(V)
        ufloat_std = sqrt(np.sum([abs(v-ufloat(mean_v,std_v))**2 for v in V])/len(V))
        
        k = ((c.k * T * ufloat(mean_v,std_v) * 1e-30) / ((ufloat_std * 1e-30)**2)) * 1e-9 # GPa
        
        mean_V.append(mean_v)
        std_V.append(std_v)
        K.append(k)
    
    #============================================================================#
    
    fig = plt.figure()
    plt.errorbar([p/max(P) for p in P], [k.nominal_value for k in K], yerr=[k.s for k in K], capsize=7.0, color="blue", linestyle="none", marker="o")
    plt.xlabel("p/p$_0$")
    plt.ylabel(r"K$_{\rm T}$ (GPa)")
    plt.title(f"{H} nm, T = 119.6 K", loc="right")
    plt.savefig(f"C:/Argon_Carbon/{H}nm/K_vs_P.pdf", bbox_inches="tight")
    plt.show(fig)
    plt.close(fig)
    
    #============================================================================#
    
    if True:
    
        file_name = f"C:/Argon_Carbon/{H}nm/{H}nm_Data.csv"
    
        with open(file_name, "w", newline="") as csvfile:
    
            csvwriter = csv.writer(csvfile)
    
            Header2 = ["Pressure (Pa)", "Bulk Modulus (GPa)", "Uncertainty (GPa)", "Mean Volume (A^3)", "Std Volume (A^3)"]
            csvwriter.writerow(Header2)
    
            for p,k,k_std,mean_v,std_v in zip(P, [k.nominal_value for k in K], [k.s for k in K], mean_V, std_V):
    
                Row = [p, k, k_std, mean_v, std_v]
                csvwriter.writerow(Row)

#============================================================================#

file2 = "C:/Argon_Carbon/2nm/2nm_Data.csv"
Data2 = np.loadtxt(file2, delimiter=",", skiprows=1)

file3 = "C:/Argon_Carbon/3nm/3nm_Data.csv"
Data3 = np.loadtxt(file3, delimiter=",", skiprows=1)

file4 = "C:/Argon_Carbon/4nm/4nm_Data.csv"
Data4 = np.loadtxt(file4, delimiter=",", skiprows=1)

file5 = "C:/Argon_Carbon/5nm/5nm_Data.csv"
Data5 = np.loadtxt(file5, delimiter=",", skiprows=1)

#============================================================================#

file2g = "C:/Argon_Carbon_Isotherms/2nm/Parallel/2nm_Data.csv"
Data2g = np.loadtxt(file2g, delimiter=",", skiprows=1)

file3g = "C:/Argon_Carbon_Isotherms/3nm/Parallel/3nm_Data.csv"
Data3g = np.loadtxt(file3g, delimiter=",", skiprows=1)

file4g = "C:/Argon_Carbon_Isotherms/4nm/Parallel/4nm_Data.csv"
Data4g = np.loadtxt(file4g, delimiter=",", skiprows=1)

file5g = "C:/Argon_Carbon_Isotherms/5nm/Parallel/5nm_Data.csv"
Data5g = np.loadtxt(file5g, delimiter=",", skiprows=1)

#============================================================================#

K_Bulk = []

Pressure_atm = np.linspace(11.71,12,100)

Pressure_Pa = [p*101325 for p in Pressure_atm]

for p in Pressure_Pa:
    
    k = 1/PropsSI("isothermal_compressibility", "P", p,"T", T, "Argon")
    
    K_Bulk.append(k)

#============================================================================#

fig = plt.figure()
plt.errorbar([p/max(Data2[:,0]) for p in Data2[:,0]], Data2[:,1], yerr=Data2[:,2], capsize=7.0, color="k", linestyle="--", marker="*", label="2 nm")
plt.errorbar([p/max(Data3[:,0]) for p in Data3[:,0]], Data3[:,1], yerr=Data3[:,2], capsize=7.0, color="red", linestyle="--", marker="v", label="3 nm")
plt.errorbar([p/max(Data4[:,0]) for p in Data4[:,0] if Data4[:,0].tolist().index(p) not in [0,1,2,3]], [k for k in Data4[:,1] if Data4[:,1].tolist().index(k) not in [0,1,2,3]], yerr=[err for err in Data4[:,2] if Data4[:,2].tolist().index(err) not in [0,1,2,3]], capsize=7.0, color="blue", linestyle="--", marker="o", label="4 nm")
plt.errorbar([p/max(Data5[:,0]) for p in Data5[:,0] if Data5[:,0].tolist().index(p) not in [0,1,2,3,4,5]], [k for k in Data5[:,1] if Data5[:,1].tolist().index(k) not in [0,1,2,3,4,5]], yerr=[err for err in Data5[:,2] if Data5[:,2].tolist().index(err) not in [0,1,2,3,4,5]], capsize=7.0, color="green", linestyle="--", marker="d", label="5 nm")
plt.plot([p/max(Pressure_atm) for p in Pressure_atm], [k*1e-9 for k in K_Bulk], color="k", linestyle="--", label="Bulk")
plt.xlabel("P (atm)")
plt.ylabel(r"K$_{\rm T}$ (GPa)")
plt.title("Argon in Carbon Slits", loc="left")
plt.title("T = 119.6 K", loc="right")
plt.legend(loc="best")
plt.savefig("C:/Argon_Carbon/K_vs_P_all.pdf", bbox_inches="tight")
plt.show(fig)
plt.close(fig)

# fig = plt.figure()

# plt.errorbar(Data2[:,0], Data2[:,1], yerr=Data2[:,2], capsize=7.0, color="k", linestyle="--", marker="*", label="2 nm NPT")
# plt.errorbar(Data3[:,0], Data3[:,1], yerr=Data3[:,2], capsize=7.0, color="red", linestyle="--", marker="v", label="3 nm NPT")
# plt.errorbar([p for p in Data4[:,0] if Data4[:,0].tolist().index(p) not in [0,1,2,3]], [k for k in Data4[:,1] if Data4[:,1].tolist().index(k) not in [0,1,2,3]], yerr=[err for err in Data4[:,2] if Data4[:,2].tolist().index(err) not in [0,1,2,3]], capsize=7.0, color="blue", linestyle="--", marker="o", label="4 nm NPT")
# plt.errorbar([p for p in Data5[:,0] if Data5[:,0].tolist().index(p) not in [0,1,2,3,4,5]], [k for k in Data5[:,1] if Data5[:,1].tolist().index(k) not in [0,1,2,3,4,5]], yerr=[err for err in Data5[:,2] if Data5[:,2].tolist().index(err) not in [0,1,2,3,4,5]], capsize=7.0, color="green", linestyle="--", marker="d", label="5 nm NPT")

# plt.errorbar(Data2[:,0], Data2g[:,4], yerr=Data2g[:,5], alpha=0.5, capsize=7.0, color="k", linestyle="--", marker="*", label="2 nm GCMC")
# plt.errorbar(Data3[:,0], Data3g[:,4], yerr=Data3g[:,5], alpha=0.5, capsize=7.0, color="red", linestyle="--", marker="v", label="3 nm GCMC")
# plt.errorbar([p for p in Data4[:,0] if Data4[:,0].tolist().index(p) not in [0,1,2,3]], [k for k in Data4g[:,4] if Data4g[:,4].tolist().index(k) not in [0,1,2,3]], yerr=[err for err in Data4g[:,5] if Data4g[:,5].tolist().index(err) not in [0,1,2,3]], alpha=0.5, capsize=7.0, color="blue", linestyle="--", marker="o", label="4 nm GCMC")
# plt.errorbar([p for p in Data5[:,0] if Data5[:,0].tolist().index(p) not in [0,1,2,3,4,5]], [k for k in Data5g[:,4] if Data5g[:,4].tolist().index(k) not in [0,1,2,3,4,5]], yerr=[err for err in Data5g[:,5] if Data5g[:,5].tolist().index(err) not in [0,1,2,3,4,5]], alpha=0.5, capsize=7.0, color="green", linestyle="--", marker="d", label="5 nm GCMC")

# plt.plot(Pressure_atm, [k*1e-9 for k in K_Bulk], color="k", linestyle="--", label="Bulk")
# plt.xlabel("P (atm)")
# plt.ylabel(r"K$_{\rm T}$ (GPa)")
# plt.title("Argon in Carbon Slits", loc="left")
# plt.title("T = 119.6 K", loc="right")
# plt.legend(loc="best")
# plt.savefig("C:/Argon_Carbon/K_vs_P_gcmc_vs_NPT.pdf", bbox_inches="tight")
# plt.show(fig)
# plt.close(fig)

fig = plt.figure()
plt.errorbar([1/h for h in [2,3,4,5]], [Data2[-1,1], Data3[-1,1], Data4[-1,1], Data5[-1,1]], yerr=[Data2[-1,2], Data3[-1,2], Data4[-1,2], Data5[-1,2]], capsize=7.0, color="blue", linestyle="--", marker=".", label="Confined")
plt.axhline(K_Bulk[-1]*1e-9, color="k", linestyle="--", label="Bulk")
plt.xlabel(r"H$^{-1}$ (nm$^{-1}$)")
plt.ylabel(r"K$_{\rm T}$ (GPa)")
plt.title("Argon in Carbon Slits", loc="left")
plt.title(r"T = 119.6 K, p/p$_0$ = 1.0", loc="right")
plt.legend(loc="best")
plt.savefig("C:/Argon_Carbon/K_vs_H_inv.pdf", bbox_inches="tight")
plt.show(fig)
plt.close(fig)






