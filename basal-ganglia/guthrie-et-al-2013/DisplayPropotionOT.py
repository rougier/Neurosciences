import numpy as np
import matplotlib.pyplot as plt

file = "PropotionOfOptimalTrials.npy"
Optimum_trials = np.load(file)

trials = np.linspace(1,121,120)
y  = 0.5 + 0.5 * (1-np.exp(-(trials - 1)/13.7)) - 0.05
  
fig = plt.figure()
axes = fig.add_subplot(1,1,1)
axes.set_autoscale_on(False)
yticks = np.linspace(0,1,11)
axes.set_xbound(0,120)
axes.set_ybound(0,1)
axes.set_yticks(yticks)
axes.plot(trials, Optimum_trials)
axes.plot(trials,y)
plt.ylabel("Proportion of optimum trials")
plt.xlabel("Trial number")
fig.savefig("PropotionOfOptimumTrials.pdf")
plt.show()