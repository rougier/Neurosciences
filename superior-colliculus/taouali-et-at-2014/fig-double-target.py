#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Wahiba Taouali (Wahiba.Taouali@inria.fr)
#               Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL
# http://www.cecill.info/index.en.html.
#
# As a counterpart to the access to the source code and rights to copy, modify
# and redistribute granted by the license, users are provided only with a
# limited warranty and the software's author, the holder of the economic
# rights, and the successive licensors have only limited liability.
#
# In this respect, the user's attention is drawn to the risks associated with
# loading, using, modifying and/or developing or reproducing the software by
# the user in light of its specific status of free software, that may mean that
# it is complicated to manipulate, and that also therefore means that it is
# reserved for developers and experienced professionals having in-depth
# computer knowledge. Users are therefore encouraged to load and test the
# software's suitability as regards their requirements in conditions enabling
# the security of their systems and/or data to be ensured and, more generally,
# to use and operate it in the same conditions as regards security.
#
# The fact that you are presently reading this means that you have had
# knowledge of the CeCILL license and that you accept its terms.
# -----------------------------------------------------------------------------
import os.path
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from model import *
from graphics import *
from stimulus import *
from parameters import *
from projections import *


# Decode function
def decode(Z, xmin=+0.0, xmax=+2.0, ymin=-1.0, ymax=+1.0,):
    Y,X = np.mgrid[0:Z.shape[0],0:Z.shape[1]]
    X = xmin + X/float(Z.shape[0]-1)*(xmax-xmin)
    Y = ymin + Y/float(Z.shape[1]-1)*(ymax-ymin)
    Z_sum = Z.sum()
    x = (Z*X).sum() / Z_sum
    y = (Z*Y).sum() / Z_sum
    return x,y

p = 100
np.random.seed(123)

if os.path.exists('double-target-5.npy'):
    T5 = np.load('double-target-5.npy')
else:
    rho = 15
    model= Model()
    T5 = np.zeros((p,2))
    for i,theta in enumerate(np.linspace(10,45,p)):
        model.reset()
        model.R = np.maximum( stimulus((rho, -theta), size=1, intensity=1) ,
                              stimulus((rho, +theta), size=1, intensity=1) )
        model.R += np.random.uniform(0,0.05,model.R.shape)

        model.run(duration=10*second, dt=5*millisecond, epsilon=0.0)
        x,y = decode(model.SC_V)
        print u"Δθ = %.2f: (%f,%f)" % (2*theta, x, y)
        T5[i] = x,y
    np.save("double-target-5.npy",T5)

if os.path.exists('double-target-10.npy'):
    T10 = np.load('double-target-10.npy')
else:
    rho = 10
    model= Model()
    T10 = np.zeros((p,2))
    for i,theta in enumerate(np.linspace(10,45,p)):
        model.reset()
        model.R = np.maximum( stimulus((rho, -theta), size=1, intensity=1) ,
                              stimulus((rho, +theta), size=1, intensity=1) )
        model.R += np.random.uniform(0,0.05,model.R.shape)

        model.run(duration=10*second, dt=5*millisecond, epsilon=0.0)
        x,y = decode(model.SC_V)
        print u"Δθ = %.2f: (%f,%f)" % (2*theta, x, y)
        T10[i] = x,y
    np.save("double-target-10.npy",T10)

if os.path.exists('double-target-15.npy'):
    T15 = np.load('double-target-15.npy')
else:
    rho = 15
    model= Model()
    T15 = np.zeros((p,2))
    for i,theta in enumerate(np.linspace(10,45,p)):
        model.reset()
        model.R = np.maximum( stimulus((rho, -theta), size=1, intensity=1) ,
                              stimulus((rho, +theta), size=1, intensity=1) )
        model.R += np.random.uniform(0,0.05,model.R.shape)

        model.run(duration=10*second, dt=5*millisecond, epsilon=0.0)
        x,y = decode(model.SC_V)
        print u"Δθ = %.2f: (%f,%f)" % (2*theta, x, y)
        T15[i] = x,y
    np.save("double-target-15.npy",T15)


fig = plt.figure(figsize=(20,7))
fig.patch.set_color('w')
G = gridspec.GridSpec(2, 7)

ax1 = plt.subplot(G[0, 0])
ax2 = plt.subplot(G[0, 1:3])
model = Model()
model.R = np.maximum( stimulus((5, -10), size=1, intensity=1) ,
                      stimulus((5, +10), size=1, intensity=1) )
model.R += np.random.uniform(0,0.05,model.R.shape)
model.R *= model.R_mask

model.run(duration=10*second, dt=5*millisecond, epsilon=0.0)
polar_frame(ax1, legend=False, labels=False)
polar_imshow(ax1, model.R)
if zoom:
    zax = zoomed_inset_axes(ax1, 6, loc=1)
    polar_frame(zax, zoom=True)
    zax.set_xlim(0.0, 0.1)
    zax.set_xticks([])
    zax.set_ylim(-.05, .05)
    zax.set_yticks([])
    zax.set_frame_on(True)
    mark_inset(ax1, zax, loc1=2, loc2=4, fc="none", ec="0.5")
    polar_imshow(zax, model.R)
logpolar_frame(ax2, legend=False, labels=False)
logpolar_imshow(ax2, model.SC_V)
ax1.text(-0.05, 1.0, 'A', va='top', ha='right',
         transform=ax1.transAxes, fontsize=20, fontweight='bold')


ax1 = plt.subplot(G[1,0])
ax2 = plt.subplot(G[1,1:3])
model = Model()
model.R = np.maximum( stimulus((5, -25), size=1, intensity=1) ,
                      stimulus((5, +25), size=1, intensity=1) )
model.R += np.random.uniform(0,0.05,model.R.shape)
model.R *= model.R_mask
model.run(duration=10*second, dt=5*millisecond, epsilon=0.0)
polar_frame(ax1, legend=False, labels=False)
polar_imshow(ax1, model.R)
if zoom:
    zax = zoomed_inset_axes(ax1, 6, loc=1)
    polar_frame(zax, zoom=True)
    zax.set_xlim(0.0, 0.1)
    zax.set_xticks([])
    zax.set_ylim(-.05, .05)
    zax.set_yticks([])
    zax.set_frame_on(True)
    mark_inset(ax1, zax, loc1=2, loc2=4, fc="none", ec="0.5")
    polar_imshow(zax, model.R)
logpolar_frame(ax2, legend=False, labels=False)
logpolar_imshow(ax2, model.SC_V)
ax1.text(-0.05, 1.0, 'B', va='top', ha='right',
         transform=ax1.transAxes, fontsize=20, fontweight='bold')

ax = plt.subplot(G[:, 3:])
X = np.linspace(20,90,p)
Y = T5[:,1]
plt.scatter( X, Y, s=50, color="g", edgecolor="g", alpha=.25)
Y = T10[:,1]
plt.scatter( X, Y, s=50, color="b", edgecolor="b", alpha=.25)
Y = T15[:,1]
plt.scatter( X, Y, s=50, color="r", edgecolor="r", alpha=.25)
plt.axvline(39, color='.75')
plt.axvline(42, color='.75')
plt.xlim(18,92)
plt.ylim(-0.5,+0.5)

plt.xlabel(u"Relative distance between targets (degrees)")
plt.ylabel(u"Normalized y position")

plt.text(20, 0, 'A',
         ha="center", va="center", size=15, fontweight='bold',
         bbox=dict(boxstyle='round', fc="w", ec="k"))

plt.text(50, +.2, 'B',
         ha="center", va="center", size=15, fontweight='bold',
         bbox=dict(boxstyle='round', fc="w", ec="k"))

plt.savefig("fig-double-target.pdf")
plt.show()
