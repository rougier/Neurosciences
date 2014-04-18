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
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

from model import *
from graphics import *
from parameters import *
from projections import *

n = 200
Rho = np.linspace(0,25,n)
print Rho

Z = np.zeros(shape=(n, colliculus_shape[1]))
shape = colliculus_shape

for i in range(len(Rho)):
     x,y = polar_to_logpolar(Rho[i]/90.0,0)
     x *= 90
     y *= 90
     Y,X = np.mgrid[0:shape[1],0:shape[0]]
     X = X / float(shape[0]) * 90
     Y = Y / float(shape[1]) * 90
     R = ((X-x)**2+(Y-y)**2)
     c = 20*stimulus_size/2.35482
     SC_V = np.exp(-R/(2*c*c))
     Z[i] = SC_V[colliculus_shape[0]/2]

     if x in [3,5,10,15]:
          model = Model()
          SC_V *= self.SV_mask
          ax = plt.subplot(111)
          logpolar_frame(ax)
          logpolar_imshow(ax, SC_V)
          plt.show()

fig = plt.figure(figsize=(10,5), facecolor='w')
for i in [3,5,10,15]:
    x,y = polar_to_logpolar( i/90.0, 0 )
    index = int(x*Z.shape[1])
    plt.plot(Rho,Z[:,index], linewidth=2) #, color='k')
    # plt.plot(X_ideal,Z_ideal[:,index], '--', linewidth=1.5, color='.5')

#plt.xlim(0.0, 25.0)
#plt.ylim(0.0,  1.1)
#plt.yticks([0.0,0.8,1.0],['0','400','500'])
#plt.vlines([5,10,15], [0,0,0], [1.1,1.1,1.1],  linewidth=1, color='.75')
#plt.xticks([2.5,10,25])
#plt.xlim(0,60)
#plt.grid()
#plt.xlabel('Target eccentricity ($^\circ$)')
#plt.ylabel('Discharge rate (spike/s)')
# # plt.savefig('tuning-curves.pdf')
plt.show()
