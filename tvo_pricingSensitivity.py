"""
Created on Wed Sep 12 13:19:41 2018

TVO pricing formula Analysis and Sensitivity
We produce ATM TVO prices on a 3D grid of dimension (9, 60, 33)
in parameter space (H, nu, rho) via all 3 pricing methods available
(MC, SVVE, DFA)

@author: Sebastian F. Tudor
"""


import numpy as np
import math
import pickle

from fSABR import fSABR
from scipy.integrate import trapz
import utils



""" TVO prices ATM for fixed maturity """
T = 1.0
N = 50000
n = math.floor(252. * T)
K = 1.0

S0   =  1.0    
sig0 =  0.1
TV   =  0.2



""" Formula analysis on a grid in parameter space """
H   = np.arange(0.05,0.49,0.05) 
nu  = np.arange(0.01,0.61,0.01) 
rho = np.arange(-0.99,0.99,0.06)


tvo_MC = np.zeros((H.shape[0], nu.shape[0], rho.shape[0]))
tvo_SVVE = np.zeros((H.shape[0], nu.shape[0], rho.shape[0]))
tvo_DFA = np.zeros((H.shape[0], nu.shape[0], rho.shape[0]))


for i in range(H.shape[0]):
    a = H[i] - 0.5
    fSV = fSABR(n, N, T, a)
    dW1 = fSV.dW1()
    dW2 = fSV.dW2()
    WH  = fSV.WH(dW1)
    for j in range(nu.shape[0]):
        sig = fSV.sig(WH, nu[j], sig0)
        for k in range(rho.shape[0]):
            dB = fSV.dB(dW1, dW2, rho[k])
            S = fSV.S(sig, dB, S0)
            ST = S[:,-1][:,np.newaxis]
            call_payoffs = np.maximum(ST - K,0)
            RV = trapz(np.square(sig),fSV.t[0,:])[:,np.newaxis]/T
            tvoCall_payoffs = TV * call_payoffs / np.sqrt(RV)
            
            tvo_MC[i,j,k] = np.mean(tvoCall_payoffs, axis = 0)[:,np.newaxis]
            tvo_SVVE[i,j,k] = utils.tvoPrice_formula(S0, sig0, K, T, TV, H[i], rho[k], nu[j])
            tvo_DFA[i,j,k] = utils.tvoPrice_formulaAlternative_3(S0, sig0, K, T, TV, H[i], rho[k], nu[j])


""" 
One can save the results in a pickle file using the following code
For plotting the results, run tvo_pricingSensitivityPlots.py
"""

with open('tvoCall_prices.pkl', 'wb') as f:  
    pickle.dump([tvo_MC, tvo_SVVE, tvo_DFA], f)       