"""
Created on Wed Sep 12 11:25:53 2018

fSABR process class

@author: Sebastian F. Tudor
"""

import numpy as np
import utils

class fSABR(object):
    """
    Class for generating paths of the fSABR model.
    """
    def __init__(self, n = 100, N = 1000, T = 1.00, a = -0.4):
        """
        Constructor for class
        """
        # Basic assignments
        self.T = T # Maturity
        self.n = n # Number of time steps
        self.dt = 1.0/self.n # Step size
        self.s = int(self.n * self.T) # Steps
        self.t = np.linspace(0, self.T, 1 + self.s)[np.newaxis,:] # Time grid
        self.a = a # Alpha = H - 0.5
        self.N = N # Paths

        # Construct hybrid scheme correlation structure
        self.e = np.array([0,0])
        self.c = utils.cov(self.a, self.n)

    def dW1(self):
        """
        Produces random numbers for variance process with required
        covariance structure.
        """
        rng = np.random.multivariate_normal
        return rng(self.e, self.c, (self.N, self.s))

    def WH(self, dW):
        """
        Constructs fractional Brownian Motion process 
        """
        Y1 = np.zeros((self.N, 1 + self.s)) # Exact integrals
        Y2 = np.zeros((self.N, 1 + self.s)) # Riemann sums

        # Construct Y1 through exact integral
        for i in np.arange(1, 1 + self.s, 1):
            Y1[:,i] += dW[:,i-1,1] 

        # Construct arrays for convolution
        G = np.zeros(1 + self.s) # Gamma
        for k in np.arange(2, 1 + self.s, 1):
            G[k] = utils.g(utils.b(k, self.a)/self.n, self.a)
            # G[k] = g2(b(k, self.a)/self.n, self.a, self.T)  

        X = dW[:,:,0] # Xi

        # Initialise convolution result, GX
        GX = np.zeros((self.N, len(X[0,:]) + len(G) - 1))

        # Compute convolution, FFT not used for small n
        # Possible to compute for all paths in C-layer?
        for i in range(self.N):
            GX[i,:] = np.convolve(G, X[i,:])

        # Extract appropriate part of convolution
        Y2 = GX[:,:1 + self.s]

        # Finally contruct and return full process
        WH = np.sqrt(2 * self.a + 1) * (Y1 + Y2)
        # WH = (Y1 + Y2) 
        return WH

    def dW2(self):
        """
        Obtain orthogonal increments
        """
        return np.random.randn(self.N, self.s) * np.sqrt(self.dt)

    def dB(self, dW1, dW2, rho = 0.0):
        """
        Constructs correlated Brownian increments dB
        """
        self.rho = rho
        dB = rho * dW1[:,:,0] + np.sqrt(1 - rho**2) * dW2
        return dB

    def sig(self, WH, nu = 0.1, sig0 = 0.1):
        """
        fSABR volatility process.
        """
        self.nu = nu
        sig = sig0 * np.exp(nu * WH)
        return sig

    def S(self, sig, dB, S0 = 1):
        """
        fSABR price process.
        """
        self.S0 = S0
        dt = self.dt

        # Construct non-anticipative Riemann increments
        increments = sig[:,:-1] * dB - 0.5 * np.square(sig[:,:-1]) * dt
        integral = np.cumsum(increments, axis = 1)

        S = np.zeros_like(sig)
        S[:,0] = S0
        S[:,1:] = S0 * np.exp(integral)
        return S