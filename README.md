# TVO pricing in the lognormal fractional SABR model

We present and test here several pricing formulas for target volatility options (TVO) assuming a lognormal fractional SABR model. The files given in this repository are organized as follows: 

* utils.py 					              Utility functions needed to implement the lognormal fSABR process
* fSABR.py				                Class implementing the fSABR process
* fSABR_test.py			              Generate and plot paths for fractional Brownian Motion and fSABR process. We also check statistical properties of fBMs
* tvo_pricing.py				          Target volatility option pricing via 3 methods: Monte Carlo and 2 analytic approximations
* tvo_pricingSensitivity.py		    TVO pricing formula analysis and sensitivity to parameter variations
* tvo_pricingSensitivityPlots.py	Plotting the sensitivities computed in tvo_pricingSensitivity.py
