import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def fitFunc(t, a, b, c):
    return a*np.exp(-b*t) + c
    
    
file = "PropotionOfOptimalTrials.npy"
Optimum_trials = np.load(file)

trials = np.linspace(1,121,120)
fitParams, fitCovariances = curve_fit(fitFunc, trials, Optimum_trials)

fig = plt.figure()
axes = fig.add_subplot(1,1,1)
axes.set_autoscale_on(False)
yticks = np.linspace(0,1,11)
axes.set_xbound(0,120)
axes.set_ybound(0,1)
axes.set_yticks(yticks)
axes.plot(trials, Optimum_trials)
axes.plot(trials, fitFunc(trials, fitParams[0], fitParams[1], fitParams[2]), "r")
plt.ylabel("Proportion of optimum trials", fontweight='bold')
plt.xlabel("Trial number", fontweight='bold')
fig.savefig("PropotionOfOptimumTrials.pdf")
plt.show()