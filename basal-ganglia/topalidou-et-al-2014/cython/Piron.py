# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt


C1 = np.load("Piron-C1.npy")[:,:60]
C2 = np.load("Piron-C2.npy")[:,:60]
C3 = np.load("Piron-C3.npy")[:,:60]
C4 = np.load("Piron-C4.npy")[:,:60]



from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

plt.figure(figsize=(10,5), dpi=72, facecolor="white")
axes = plt.subplot(111)
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.xaxis.set_ticks_position('bottom')
axes.spines['bottom'].set_position(('data',0))
axes.yaxis.set_ticks_position('left')

n = C1.shape[1]


axes.plot(np.arange(n), np.mean(C1["P"],axis=0),
          lw=1.5, c='0.5', linestyle="--", label="HC with GPi")
axes.plot(np.arange(n), np.mean(C3["P"],axis=0),
          lw=1.5, c='0.0', linestyle="--", label="NC with GPi")
axes.plot(np.arange(n), np.mean(C2["P"],axis=0),
          lw=1.5, c='0.5', linestyle="-", label="HC without GPi")
axes.plot(np.arange(n), np.mean(C4["P"],axis=0),
          lw=1.5, c='0.0', linestyle="-", label="NC without GPi")

plt.legend(loc='lower right', frameon=False)

plt.xlabel("Trial number")
plt.ylabel("Proportion of optimum trials")

plt.xlim(0,n)
plt.ylim(0,1.05)
plt.tight_layout()
plt.savefig("Performances.pdf")
plt.show()



fig = plt.figure(figsize=(6,5), dpi=72, facecolor="white")
fig.subplots_adjust(bottom=0.25)
fig.subplots_adjust(left=0.15)

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

# def label(i,j,text,X,Y):
#     x = (X[i]+X[j])/2
#     y = 1.25*max(Y[i], Y[j])
#     dx = abs(X[i]-X[j])
#     props = {'connectionstyle':'bar','arrowstyle':'-', 'shrinkA':20,'shrinkB':20,'lw':1}
#     ax.annotate(text, xy=(x,y+100), zorder=10, fontsize=20, ha="center")
#     ax.annotate('', xy=(X[i],y), xytext=(X[j],y), arrowprops=props)
# label(0,1,"***", indices+width/2, means+stds)
# label(1,2,"***", indices+width/2, means+stds)
# label(2,3,"***", indices+width/2, means+stds)

plt.ylabel("Reaction time (ms)", fontsize=16)
plt.xlim(0,4.25)
plt.ylim(0,1200)
#plt.tight_layout()

b = 0.025

plt.axhline(-125, b,.5-b, clip_on=False, color="k")
ax.text(1.125,-150,"With GPi", clip_on=False, ha="center", va="top", fontsize=18)

plt.axhline(-125, .5+b,1-b, clip_on=False, color="k")
ax.text(3.125,-150,"Without GPi", clip_on=False, ha="center", va="top", fontsize=18)

plt.savefig("RT.pdf")
plt.show()
