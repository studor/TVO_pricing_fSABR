"""
Created on Wed Sep 12 11:17:45 2018

Utility functions needed to implement the lognormal fSABR process

@author: Sebastian F. Tudor
"""

import numpy as np
import math as math
from scipy import integrate
from scipy.special import hyp2f1 
from scipy.special import beta
from scipy.stats import norm
import scipy.optimize as so

def g(x, a):
    """
    Power law type kernel 
    """
    return x**a

def g2(x,a,T):
    """
    Kernel applicable to fSABR process
    """
    H = a + 0.5
    cH = np.sqrt(2*H*math.gamma(3/2-H)/(math.gamma(2-2*H)*math.gamma(H+1/2)))
    return cH * (x**a) * hyp2f1(H-0.5, 0.5-H, H+0.5, -x/(T-x))

def b(k, a):
    """
    Optimal discretisation
    """
    return ((k**(a+1)-(k-1)**(a+1))/(a+1))**(1/a)

def cov(a, n):
    """
    Covariance matrix for given alpha and n
    """
    cov = np.array([[0.,0.],[0.,0.]])
    cov[0,0] = 1./n
    cov[0,1] = 1./((1.*a+1) * n**(1.*a+1))
    cov[1,1] = 1./((2.*a+1) * n**(2.*a+1))
    cov[1,0] = cov[0,1]
    return cov






def bs(F, K, V, o = 'call'):
    """
    Black call price for given forward, strike and integrated variance
    """
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    sv = np.sqrt(V)
    d1 = np.log(F/K) / sv + 0.5 * sv
    d2 = d1 - sv
    P = w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)
    return P

def bs_normalized(x,v):
    d1 = x/np.sqrt(v) + np.sqrt(v)/2
    d2 = d1 - np.sqrt(v)
    P = np.exp(x) * norm.cdf(d1) - norm.cdf(d2)
    return P

def bs_tvo(F, K, tau, TV, s, o = 'call'):
    """
    Returns the Black call price of a TV call option for given forward, strike,
    target volatility, and integrated variance
    """
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K>1.0) - 1
        
    d1 = np.log(F/K) / (s * np.sqrt(tau)) + 0.5 * s * np.sqrt(tau) 
    d2 = d1 - s * np.sqrt(tau)
    P = (w * F * norm.cdf(w * d1) - w * K * norm.cdf(w * d2)) * TV / s
    return P


def bsinv(P, F, K, t, o = 'call'):
    """
    Returns implied Black vol from given call price, forward, strike and time
    to maturity.
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1

    # Ensure at least instrinsic value
    P = np.maximum(P, np.maximum(w * (F - K), 0))

    def error(s):
        return (bs(F, K, s**2 * t, o) - P)
    s = so.brentq(error, 1e-9, 1e+9)
    return s


def bsinv_tvo(P, F, K, TV, t, o = 'call'):
    """
    Returns implied volatility from given TVO call price, forward, strike, 
    target volatility and time to maturity
    """
    # Set appropriate weight for option token o
    w = 1
    if o == 'put':
        w = -1
    elif o == 'otm':
        w = 2 * (K > 1.0) - 1
    
    def error2(s):
        return (bs_tvo(F, K, t, TV, s, o) - P )#np.maximum(P, np.maximum(w * (F - K)*TV/s, 0)))
    
    s = so.newton(error2, 0.08, tol=1.48e-06)
    return s






def covGamma(k, H):
    gam = ((np.abs(k+1))**(2*H) - 2.*(np.abs(k))**(2*H) + (np.abs(k-1))**(2*H))/2
    return gam

def tvoPrice_formula(S0, sig0, K, T, TV, H, rho, nu):
    S0 = np.log(S0/K)
    cH = np.sqrt(2*H*math.gamma(3/2-H)/(math.gamma(2-2*H)*math.gamma(H+1/2)))
    rhob = np.sqrt(1-rho**2)
    pi = math.pi
    kappa1 = nu*TV*K *2*cH*T**H * rhob/(np.sqrt(2*pi)*(2*H+3))
    kappa2 = nu*TV*K *2*cH*T**(H-1/2)/(sig0*(2*H+3))
    kappa3 = nu*TV*K*rho**2*sig0* 2*cH*T**(H+1/2)/(2*H+3) + nu*TV*K *2*cH*T**(H-1/2) /(sig0*(2*H+3))
    kappa4 = nu*TV*K*rho* 2*cH*T**(H+1/2)/(2*H+3)
    kappa5 = nu*TV*K*rho* 2*cH*T**(H+3/2)/(2*H+3)

    a  = rhob*sig0*np.sqrt(T);
    s2 = (rho/rhob)**2;
    N1 = S0/sig0/np.sqrt(T) + sig0*np.sqrt(T)/2
    N2 = S0/sig0/np.sqrt(T) - sig0*np.sqrt(T)/2
    F1 = np.exp(S0)*norm.cdf(N1)
    F2 = norm.cdf(N2)
    mud1 = S0/sig0/np.sqrt(T)/rhob + sig0*np.sqrt(T)/2 * (1-2*rho**2)/rhob
    mud2 = N2/rhob
    aux5 = np.exp(a*mud1+a**2*s2/2) * ( ((mud1+a*s2)*(mud1+a*s2/2)+s2/2) * norm.pdf(N1) + s2/np.sqrt(s2+1)* (2*mud1 + 3*s2*a**2/2)*norm.pdf(N1) - s2**2/(1+s2) * N1*norm.pdf(N1))
    
    E1 = rhob*rho*N2*np.sqrt(T)*np.exp(N2*(1/T - rho**2*N2))
    E2 = N2*np.sqrt(T)/rho * norm.cdf(N2) + rho*np.sqrt(T)*norm.pdf(N2) - rhob*np.sqrt(T)/rho * mud2*F2
    E3 = rhob*np.sqrt(T)/rho * (np.exp(rhob*S0 - a**2/2 + a**2/2/rhob) *(N1*norm.cdf(N1) + rho**2/rhob*norm.pdf(N2)) - mud1*F1)
    E4 = rhob**2*T/rho**2 * (np.exp(a**2/2)*aux5 - 2*mud1*rho/rhob/np.sqrt(T)*E3 - mud1**2*F1)
    E5 = norm.cdf(N1)*np.exp(S0);
    
    tvo_formula = TV*K/sig0 * bs_normalized(S0,(sig0**2)*T) + kappa1*E1 + kappa2*E2 - kappa3*E3 + kappa4*E4 - kappa5*E5
    return tvo_formula



""" 
Utilities used to compute the double integral for TVO price via 
alternative formula given in Corollary 1
"""

def Kernel(t,s,H):
    """Kernel function"""
    cH = np.sqrt(2*H*math.gamma(3/2-H)/(math.gamma(2-2*H)*math.gamma(H+1/2)))
    """if s>t: 
        KH = 0
    else:"""
    KH = cH * (t-s)**(H-0.5) * hyp2f1(H-0.5, 0.5-H, H+0.5, (s-t)/s)
       
    return KH

def integrandKernel(s,r,tau,H):
    return Kernel(tau,s,H)*Kernel(r,s,H)

def integrandGeneral(r, tau, nu, H, T):
    I  = integrate.quad(integrandKernel, 0.00001, tau, args = (r,tau,H))[0]
    IG = np.exp(nu**2/2 * (tau**(2*H) + 4*r**(2*H) + 4*I)) * Kernel(r,tau,H)
    return IG

def bounds_r(tau, nu, H, T):
    return [tau+0.0001, T]

def bounds_tau(nu, H, T):
    return [0.0001, T]

def tvoPrice_formulaAlternative(S0, sig0, K, T, TV, H, rho, nu):
    X0 = np.log(S0/K)
    M0 = sig0**2 * integrate.quad(lambda x: np.exp(2*nu**2*x**(2*H)), 0, T)[0]
    I  = integrate.nquad(integrandGeneral, [bounds_r, bounds_tau], 
                         args = (nu, H, T))[0]
    C = bs_normalized(X0,M0)
    
    d1 = X0/np.sqrt(M0) + np.sqrt(M0)/2
    d2 = d1 - np.sqrt(M0)
    Cx = np.exp(X0) * norm.cdf(d1) 
    Cxw = np.exp(-d2**2/2)/(2*np.sqrt(2*math.pi*M0)) * (1/2 - X0/M0)
    Fxwh = -Cx/2/(M0**(3/2)) + Cxw/np.sqrt(M0)
    price = 2*nu*rho*Fxwh*(sig0**3) * I  +  C/np.sqrt(M0)
    price = price * K*TV*np.sqrt(T)
    return price



""" 
Simplified and faster version of formula in Corollary 1
Formula is given in Remark 4 and two versions are implemented below
"""

def Kernel_2(t,s,H,T):
    """Kernel function"""
    cH = np.sqrt(2*H*math.gamma(3/2-H)/(math.gamma(2-2*H)*math.gamma(H+1/2)))
    KH = cH * (t-s)**(H-0.5) * hyp2f1(H-0.5, 0.5-H, H+0.5, (s-t)/s)   
    return KH

def bounds_r_2(tau, H, T):
    return [tau+0.0001, T]

def bounds_tau_2(H, T):
    return [0.0001, T]

def tvoPrice_formulaAlternative_2(S0, sig0, K, T, TV, H, rho, nu):
    X0 = np.log(S0/K)
    M0 = sig0**2 * integrate.quad(lambda x: np.exp(2*nu**2*x**(2*H)), 0, T)[0]
    I  = integrate.nquad(Kernel_2, [bounds_r_2, bounds_tau_2], args = (H,T))[0]
    C  = bs_normalized(X0,M0)
    I2 = T**(2*H+2) / (2*(2*H+2))
    
    d1 = X0/np.sqrt(M0) + np.sqrt(M0)/2
    d2 = d1 - np.sqrt(M0)
    Cx = np.exp(X0) * norm.cdf(d1) 
    Cw = np.exp(-d2**2/2)/(2*np.sqrt(2*math.pi*M0))
    Cxw = Cw * (1/2 - X0/M0)
    Cww = -Cw * (1/2/M0 + 1/8 - (X0/M0)**2/2)
    
    Fxwh  = -Cx/2/(M0**(3/2)) + Cxw/np.sqrt(M0)
    Fwhwh = -Cw/(M0**(3/2)) + 3*C/4/(M0**(5/2)) + Cww/np.sqrt(M0) 
    
    price = C/np.sqrt(M0) + 2*nu*rho*Fxwh*(sig0**3) * I  + 4*nu**2*Fwhwh*(sig0**4) * I2 
    price = price * K*TV*np.sqrt(T)
    return price


def tvoPrice_formulaAlternative_3(S0, sig0, K, T, TV, H, rho, nu):
    X0 = np.log(S0/K)
    M0 = sig0**2 * integrate.quad(lambda x: np.exp(2*nu**2*x**(2*H)), 0, T)[0]
    
    cH = np.sqrt(2*H*math.gamma(3/2-H)/(math.gamma(2-2*H)*math.gamma(H+1/2)))
    kappaH = cH * beta(3/2-H, H+1/2) / (H+1/2) 
    I  = kappaH * T**(3/2+H) / (3/2+H)
    
    C  = bs_normalized(X0,M0)
    I2 = T**(2*H+2) / (2*(2*H+2))
    
    d1 = X0/np.sqrt(M0) + np.sqrt(M0)/2
    d2 = d1 - np.sqrt(M0)
    Cx = np.exp(X0) * norm.cdf(d1) 
    Cw = np.exp(-d2**2/2)/(2*np.sqrt(2*math.pi*M0))
    Cxw = Cw * (1/2 - X0/M0)
    Cww = -Cw * (1/2/M0 + 1/8 - (X0/M0)**2/2)
    
    Fxwh  = -Cx/2/(M0**(3/2)) + Cxw/np.sqrt(M0)
    Fwhwh = -Cw/(M0**(3/2)) + 3*C/4/(M0**(5/2)) + Cww/np.sqrt(M0) 
    
    price = C/np.sqrt(M0) + 2*nu*rho*Fxwh*(sig0**3) * I  + 4*nu**2*Fwhwh*(sig0**4) * I2 
    price = price * K*TV*np.sqrt(T)
    return price
