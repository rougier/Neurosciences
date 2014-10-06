import numpy as np
import matplotlib.pyplot as plt

folder = "Results/simulation_"
array = "/Cue_Rw_Choice.npy"
Optimum_trials = np.zeros(120)
for i in range(0,250,1):
    file = folder + str(i+1) + array
    load = np.load(file)
    Optimum_trials = Optimum_trials + load[:,6]

Optimum_trials = Optimum_trials/250.0
file = 'PropotionOfOptimalTrials.npy'
np.save(file, Optimum_trials)

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