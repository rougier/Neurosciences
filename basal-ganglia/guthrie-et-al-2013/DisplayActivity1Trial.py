import numpy as np
import matplotlib.pyplot as plt
# Display Cortex's Activity from a Trial    

folder = "Results/simulation_"
cog = "/Cortex_cog.npy"
mot = "/Cortex_mot.npy"
file_cog = folder + str(1) + cog
file_mot = folder + str(1) + mot

Cortex_cog = np.load(file_cog)
Cortex_mot = np.load(file_mot)

trial_time = 3 * 1000 
duration = np.linspace(trial_time*0,trial_time*1-1,3000).astype(int)

C = np.array(Cortex_cog)
M = np.array(Cortex_mot)
timesteps = np.linspace(0.,3.,len(C[duration]))
plt.plot(timesteps, C[duration,0],'b', label = 'Cognitive Cortex')
plt.plot(timesteps, C[duration,1],'b')
plt.plot(timesteps, C[duration,2],'b')
plt.plot(timesteps, C[duration,3],'b')
plt.plot(timesteps, M[duration,0],'r', label = 'Motor Cortex')
plt.plot(timesteps, M[duration,1],'r')
plt.plot(timesteps, M[duration,2],'r')
plt.plot(timesteps, M[duration,3],'r')
plt.xlabel("Time (seconds)")
plt.ylabel("Activity (Hz)")
plt.legend(frameon=False, loc='upper left')
plt.xlim(0.0,2.5)
plt.ylim(-10.0,80.0)

plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
           ['0.0','0.5\n(Trial start)','1.0','1.5', '2.0','2.5\n(Trial stop)','3.0'])
plt.show()