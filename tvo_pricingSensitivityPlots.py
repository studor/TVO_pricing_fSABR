"""
Created on Wed Sep 12 13:38:20 2018

Run this file for plotting purposes

@author: Sebastian F. Tudor
"""

import numpy as np
import pickle
import mpl_toolkits.mplot3d
from matplotlib import pyplot as plt
from matplotlib import cm
from matplotlib.ticker import FormatStrFormatter
 




H   = np.arange(0.05,0.49,0.05) 
nu  = np.arange(0.01,0.61,0.01) 
rho = np.arange(-0.99,0.99,0.06)

with open('tvoCall_prices.pkl', 'rb') as f:
    tvo_MC, tvo_SVVE, tvo_DFA = pickle.load(f)
    

tvo_formula = tvo_SVVE
tvoCall_prices = tvo_MC
errFormula = np.divide(tvoCall_prices - tvo_formula, tvoCall_prices)


"""
Keeping H fixed
max(coefH) = 9
"""

coefH = 4
X, Y = np.meshgrid(rho,nu)
errFormulaH = errFormula[coefH,:,:]

fig = plt.figure()
ax = fig.gca(projection='3d')

ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
surf = ax.plot_surface(X, Y, errFormulaH, cmap=cm.coolwarm, linewidth=0, 
                       antialiased=False)
title = r'Relative Error for H = %.2f'%H[coefH]
ax.set_title(title)
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\nu$')
ax.set_zlabel('Relative Error')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
surf = ax.plot_surface(X, Y, tvo_formula[coefH,:,:], cmap=cm.coolwarm, 
                       linewidth=0, antialiased=False)
title = r'TVO prices for H = %.2f'%H[coefH]
ax.set_title(title)
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$\nu$')
ax.set_zlabel('TVO Calls')








"""
Keeping \nu fixed
max(coefNu) = 60
"""

coefNu = 10
X, Y = np.meshgrid(rho,H)
errFormulaNu = errFormula[:,coefNu,:]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_ticks(np.arange(0, 0.55, 0.1))
surf = ax.plot_surface(X, Y, errFormulaNu, cmap=cm.coolwarm, 
                       linewidth=0, antialiased=False)
title = r'Relative Error for $\nu$ = %.2f'%nu[coefNu]
ax.set_title(title)
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'$H$')
ax.set_zlabel('Relative Error')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_ticks(np.arange(0, 0.55, 0.1))
surf = ax.plot_surface(X, Y, tvo_formula[:,coefNu,:], cmap=cm.coolwarm, 
                       linewidth=0, antialiased=False)
title = r'TVO prices for $\nu$ = %.2f'%nu[coefNu]
ax.set_title(title)
ax.set_xlabel(r'$\rho$')
ax.set_ylabel(r'H')
ax.set_zlabel('TVO Calls')






"""
Keeping \rho fixed
max(coefRho) = 33
"""

coefRho = 11
X, Y = np.meshgrid(nu,H)
errFormulaRho = errFormula[:,:,coefRho]

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_ticks(np.arange(0, 0.55, 0.1))
surf = ax.plot_surface(X, Y, errFormulaRho, cmap=cm.coolwarm, linewidth=0, 
                       antialiased=False)
title = r'Relative Error for $\rho$ = %.2f'%rho[coefRho]
ax.set_title(title)
ax.set_xlabel(r'$\nu$')
ax.set_ylabel(r'$H$')
ax.set_zlabel('Relative Error')

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.xaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
ax.yaxis.set_ticks(np.arange(0, 0.55, 0.1))
surf = ax.plot_surface(X, Y, tvo_formula[:,:,coefRho], cmap=cm.coolwarm, 
                       linewidth=0, antialiased=False)
title = r'TVO prices for $\rho$ = %.2f'%rho[coefRho]
ax.set_title(title)
ax.set_xlabel(r'$\nu$')
ax.set_ylabel(r'H')
ax.set_zlabel('TVO Calls')    