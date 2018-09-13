"""
Created on Wed Sep 12 12:56:22 2018

Traget Volatility Option pricing

We compare here 3 ways for computing the TVO price:
    Monte Carlo simulation
    Small volatility of volatility expansion (first order approximation)
    Approximation formula via Malliavin calculus (first order approximation)

@author: Sebastian F. Tudor
"""

import numpy as np
import time 
import utils

from fSABR import fSABR
from scipy.integrate import trapz
from matplotlib import pyplot as plt


T = 1.0
H = 0.1


nu = 0.05
S0 = 1.0
sig0 = 0.3
rho = -0.7

TV = 0.3
k = np.arange(-0.22, 0.22, 0.01)
K = np.exp(k)[np.newaxis,:]/S0


""" TVO pricing via DFA (4.7) """
startTimer = time.time()
price1  = utils.tvoPrice_formulaAlternative_3(S0, sig0, K, T, TV, H, rho, nu)
endTimer = time.time()
t1 = endTimer - startTimer 


""" TVO pricing via SVVE (5.6) """
startTimer = time.time()
price2 = utils.tvoPrice_formula(S0, sig0, K, T, TV, H, rho, nu)
endTimer = time.time()
t2 = endTimer - startTimer



""" TVO pricing via MC """
n = 1000
N = 50000
a = H - 0.5

fSV = fSABR(n, N, T, a)
dW1 = fSV.dW1()
dW2 = fSV.dW2()
dB = fSV.dB(dW1, dW2, rho)

WH  = fSV.WH(dW1)
sig = fSV.sig(WH, nu, sig0)
S = fSV.S(sig, dB, S0)


startTimer = time.time()
ST = S[:,-1][:,np.newaxis]
call_payoffs = np.maximum(ST - K,0)
RV = trapz(np.square(sig),fSV.t[0,:])[:,np.newaxis]/T
tvoCall_payoffs = TV * call_payoffs / np.sqrt(RV)
price3 = np.mean(tvoCall_payoffs, axis = 0)[:,np.newaxis]
endTimer = time.time()
t3 = endTimer - startTimer



""" 
We have the following notations:
    price1 - Decomposition Formula Approx. (DFA)
    price2 - Small vol of vol expansion (SVVE)
    price3 - MC simulation
"""

plot, axes = plt.subplots()
axes.plot(np.transpose(K), np.transpose(price1), 'r')
axes.plot(np.transpose(K), np.transpose(price2), 'b')
axes.plot(np.transpose(K), price3, 'g--')
axes.set_xlabel(r'$K/S_0$', fontsize=12)
axes.set_ylabel(r'TVO Call Price', fontsize=12)
axes.legend(['Decomposition Formula Approximation',
             'Small Vol of Vol Expansion','Monte Carlo Simulation'])
title = r'$T=%.2f,\ H=%.2f,\ \rho=%.2f,\ \nu=%.2f,\ \sigma_0=%.2f,\ \bar\sigma=%.2f$'
axes.set_title(title%(fSV.T, H, fSV.rho, fSV.nu, sig0 , TV), fontsize=12)
plt.grid(True)


# Concatenate arrays for Latex purposes
err1 = np.divide(np.absolute(price3 - np.transpose(price1)), price3) * 100
err2 = np.divide(np.absolute(price3 - np.transpose(price2)), price3) * 100

tableTex = np.concatenate((np.transpose(K), price3), axis = 1)
tableTex = np.concatenate((tableTex, np.transpose(price1)), axis = 1)
tableTex = np.concatenate((tableTex, np.transpose(price2)), axis = 1)

tableTex = np.concatenate((tableTex, err1), axis = 1) 
tableTex = np.concatenate((tableTex, err2), axis = 1) 

# Computing times
print('Computing times in seconds:')
print('SVEE = %.5f s; '%t1, 'DFA = %.5f s; '%t2, 'MC = %.5f s; '%t3)

