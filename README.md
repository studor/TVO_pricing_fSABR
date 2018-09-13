# TVO pricing in the lognormal fractional SABR model

We present and test here several pricing formulas for target volatility options (TVO) assuming a lognormal fractional SABR (fSABR) model, having the stochastic volatility driven by a fractional Brownian motion (fBM). The formulas were developed and proved in the paper [Alos et al. (2018)](https://arxiv.org/pdf/1801.08215.pdf) available on arXiv.


The files given in this repository are organized as follows: 

File                            | Description
--------------------------------|------------------------------------------------------------------------------------------ 
tvo_pricingNotebook.ipynb       | Example Jupyter which demonstrates usage of the code
utils.py 					              | Utility functions needed to implement the lognormal fSABR process
fSABR.py				                | Class implementing the fSABR process
fSABR_test.py			              | Generate paths for fBM and fSABR process; check statistical properties of fBMs
tvo_pricing.py				          | TVO pricing via 3 methods: Monte Carlo and 2 analytic approximations
tvo_pricingSensitivity.py		    | TVO pricing formulae analysis and sensitivity to parameter variations
tvo_pricingSensitivityPlots.py	| Plotting the sensitivities computed in tvo_pricingSensitivity.py


Tested with Python 3.6.3 (Anaconda custom on 64-bit) on a macOS X Yosemite (version 10.10.5) with processor 2 x 2.4 GHz 6-Core Intel Xeon.
