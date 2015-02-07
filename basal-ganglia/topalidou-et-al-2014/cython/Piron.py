# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


C1 = np.load("Piron-C1.npy")
C2 = np.load("Piron-C2.npy")
C3 = np.load("Piron-C3.npy")
C4 = np.load("Piron-C4.npy")


from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

plt.figure(figsize=(8,4), dpi=72, facecolor="white")
axes = plt.subplot(111)
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.xaxis.set_ticks_position('bottom')
axes.spines['bottom'].set_position(('data',0))
axes.yaxis.set_ticks_position('left')



axes.plot(np.arange(120), np.mean(C1["P"],axis=0),
          lw=1.5, c='0.5', linestyle="--", label="HC with GPi")
axes.plot(np.arange(120), np.mean(C3["P"],axis=0),
          lw=1.5, c='0.0', linestyle="--", label="NC with GPi")
axes.plot(np.arange(120), np.mean(C2["P"],axis=0),
          lw=1.5, c='0.5', linestyle="-", label="HC without GPi")
axes.plot(np.arange(120), np.mean(C4["P"],axis=0),
          lw=1.5, c='0.0', linestyle="-", label="HC without GPi")

plt.legend(loc='lower right', frameon=False)

plt.xlabel("Trial number")
plt.ylabel("Proportion of optimum trials")

plt.xlim(0,119)
plt.ylim(0,1.05)
plt.tight_layout()
plt.savefig("Performances.pdf")
plt.show()



plt.figure(figsize=(6,5), dpi=72, facecolor="white")
ax = plt.subplot(111)
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data',0))
ax.yaxis.set_ticks_position('left')

means = [np.mean(C1["RT"]), np.mean(C3["RT"]), np.mean(C2["RT"]), np.mean(C4["RT"])]
stds  = [ np.std(C1["RT"]),  np.std(C3["RT"]),  np.std(C2["RT"]),  np.std(C4["RT"])]

indices = 0.25+np.arange(4)
width=0.75
p1 = plt.bar(indices, means, width=width,  yerr=stds,
             color=["1.", ".5", "1.", ".5"], edgecolor='k', ecolor='k')
plt.xticks(indices+width/2., ('HC', 'NC', 'HC', 'NC') )

plt.ylabel("Reaction time (ms)", fontsize=16)
plt.xlim(0,4.25)
plt.ylim(0,1100)
plt.tight_layout()
plt.savefig("RT.pdf")
plt.show()
