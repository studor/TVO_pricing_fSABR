"""
Created on Wed Sep 12 11:29:13 2018

fBM and fSABR processes paths and properties 

@author: Sebastian F. Tudor
"""

import numpy as np
import scipy as sp
import utils
from fSABR import fSABR

from numpy import linalg
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter



""" 
Process parameters

Some notations:
    N       number of Monte Carlo paths
    n       number of time steps
    T       simulation time interval is [0,T] 
    H       Hurst parameter
    rho     covariance
    S0      price process at time t = 0
    sig0    fractional stochastic vol process at time t = 0
    
"""

N = 50000
n = 1000
T = 1.0

H    =  0.2
rho  =  -0.7
nu   =  0.1
S0   =  1.0
sig0 =  0.2


a = H - 0.5

""" fSABR process paths """
fSV = fSABR(n, N, T, a)

dW1 = fSV.dW1()
dW2 = fSV.dW2()
dB = fSV.dB(dW1, dW2, rho)

WH  = fSV.WH(dW1)
sig = fSV.sig(WH, nu, sig0)
S = fSV.S(sig, dB, S0)




""" Plotting some sample paths """

plt.title('fBM with H = %.1f'%H)
plt.xlabel('t')
plt.ylabel('$W^H_t$')
plt.plot(fSV.t[0,:], WH[0,:], 'g')
plt.grid(True); plt.show()

plt.title('Fractional SV')
plt.xlabel('t')
plt.ylabel('$\sigma_t$')
plt.plot(fSV.t[0,:], sig[0,:], 'g')
plt.grid(True); plt.show()

plt.title('Price process in fSABR')
plt.xlabel('t')
plt.ylabel('$S_t$')
plt.plot(fSV.t[0,:], S[0,:], 'g')
plt.grid(True); plt.show(); 




""" Check Statistical Properties of the fBM via MC """

eY1 = 0 * fSV.t                                         # Known expectation
vY1 = fSV.t**(2*fSV.a + 1)                              # Known variance
eY2 = np.mean(WH, axis=0, keepdims=True)                # Observed expectation
vY2 = np.var(WH, axis=0, keepdims=True)                 # Observed variance

plt.plot(fSV.t[0,:], eY1[0,:], 'r')
plt.plot(fSV.t[0,:], eY2[0,:], 'g') 
plt.xlabel(r'$t$')
plt.ylabel(r'$E[W^H_t]$')
plt.title(r'Expected value of simulated fBM for N = %d paths'%N)
plt.grid(True); plt.show()

plt.plot(fSV.t[0,:], vY1[0,:], 'r')
plt.plot(fSV.t[0,:], vY2[0,:], 'g') 
plt.xlabel(r'$t$')
plt.ylabel(r'$Var(W^H_t)$')
plt.title(r'Variance of simulated fBM for N = %d paths'%N)
plt.legend(['$t^{2H}$','Monte Carlo'])
plt.grid(True); plt.show()








""" Check the 2D covariance structure of the simulated fBM via MC """

# Make the data
X, Y = np.meshgrid(fSV.t, fSV.t)
Z  = 0.5* (X**(2*H) + Y**(2*H) - (np.abs(X-Y))**(2*H))

# Compute covariance structure of simulated fBM via MC
Z2 = np.cov(WH, rowvar = False)

# Compute error
err = linalg.norm(Z-Z2)
errSurf = Z-Z2 

# Plot covariance surface for verification
fig = plt.figure()
ax = fig.gca(projection='3d')
surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, linewidth=0, 
                       antialiased=False)
ax.set_zlim(-1.01, 1.51)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.01f'))
title = r'Covariance function $\gamma(s,t)$'
ax.set_title(title, fontsize=16)
ax.set_xlabel(r't')
ax.set_ylabel(r's')
ax.set_zlabel('$\gamma$')
plt.show()

# Plot error surface
fig = plt.figure()
ax = fig.gca(projection='3d')
surf3 = ax.plot_surface(X, Y, errSurf, 
                        cmap=cm.coolwarm, linewidth=0, antialiased=False)
ax.set_zlim(-0.05, 0.1)
ax.zaxis.set_major_locator(LinearLocator(10))
ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))
title = r'Absolute error surface'
ax.set_title(title, fontsize=16)
ax.set_xlabel(r't')
ax.set_ylabel(r's')
plt.show()







""" 
Hypothesis testing
NULL hypotheses: 
    the covariances of the sample are in accordance with fractional 
    Gaussian noise for some specified Hurst parameter H

We use a chi-square test for fractional Gaussian noise
Test: reject NULL hypothesis when CN < chi2Test 
"""

XH = np.diff(WH)
Gam = [[utils.covGamma(i-j,H) for i in range(n)] for j in range(n)]
L  = linalg.cholesky(Gam)
ZH = (linalg.inv(L)).dot(np.transpose(XH))
CN = (linalg.norm(ZH, 2))**2            # Test statistic

alpha = 0.99                            # Confidence level
chi2Test = sp.stats.chi2.ppf(alpha,n)   # p value of the chi2Test
print('Reject null hypothesis: ', CN<chi2Test)