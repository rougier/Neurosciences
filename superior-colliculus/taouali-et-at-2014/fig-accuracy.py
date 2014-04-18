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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, zoomed_inset_axes

from model import *
from graphics import *
from stimulus import *
from parameters import *
from projections import *


def decode(Z, xmin=+0.0, xmax=+2.0, ymin=-1.0, ymax=+1.0,):
    Y,X = np.mgrid[0:Z.shape[0],0:Z.shape[1]]
    X = xmin + X/float(Z.shape[0]-1)*(xmax-xmin)
    Y = ymin + Y/float(Z.shape[1]-1)*(ymax-ymin)
    Z_sum = Z.sum()
    x = (Z*X).sum() / Z_sum
    y = (Z*Y).sum() / Z_sum
    return x,y

targets = []
for i in [2,3,4,5,6,7,8,9,10,15,20]:
    for j in [-45, -30, -15, 0, +15, +30, +45]:
        targets.append((i,j))

T = np.zeros((len(targets),2))
for i,target in enumerate(targets):
    rho,theta = target
    rho,theta  = rho/90.0, np.pi*theta/180.0
    x,y = polar_to_logpolar(rho,theta)
    T[i] = 2*x,2*y-1


if not os.path.exists('accuracy.npy'):
    model = Model()
    D = np.zeros((len(targets),2))
    for i,target in enumerate(targets):
        rho,theta = target
        x_,y_ = polar_to_logpolar(rho/90.0, np.pi*theta/180.0)
        x_,y_ = 2*x_, 2*y_-1
        model.reset()
        model.R = stimulus((rho, theta))
        model.run(duration=5*second, dt=5*millisecond, epsilon=0.001)
        x,y = decode(model.SC_V)
        D[i] = x,y
        print 'Target at (%d,%d): %f' % (rho,theta, np.sqrt((x-x_)*(x-x_) + (y-y_)*(y-y_)))
    np.save('accuracy.npy', D)
else:
    D = np.load('accuracy.npy')


fig = plt.figure(figsize=(8,8), facecolor='w')
ax1 = plt.subplot(111, aspect=1)
logpolar_frame(ax1)
for i in range(len(D)):
    plt.plot([T[i,0],D[i,0]],[T[i,1],D[i,1]], color='k')
ax1.scatter(T[:,0], T[:,1], s=50, color='k', marker='o')
ax1.scatter(D[:,0], D[:,1], s=50, color='k', marker='o', facecolors='w')

#axins = inset_axes(ax1, width='25%', height='8%', loc=3)
#X = np.linspace(0,90,model.RT_shape[1])
#retina = stimulus(position=(45.0,0.0), size= 25/90.0)
#axins.plot(X, retina[retina_shape[0]/2], lw=1, color='k')
#axins.set_xticks([])
#axins.set_yticks([])
#axins.set_xlim(25,65)
#axins.set_ylim(-0.1,1.1)
plt.savefig("fig-accuracy.pdf")
plt.show()
