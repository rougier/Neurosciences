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
from projections import *


model = Model()

p = 200
X = np.linspace(0,25,p)
Z = np.zeros(shape=(p, colliculus_shape[1]))

# size = 2.0 °, intensity = 0.5
if 0 or not os.path.exists('tuning.npy'):
     for i in range(p):
         model.reset()
         model.R = stimulus((X[i], 0.0), size=2.0, intensity=0.5)
         model.run(duration=20*second, dt=5*millisecond, epsilon=0.001)
         Z[i] = model.SC_V[colliculus_shape[0]/2]
         print "%d/%d: %f" %  (i,p, Z[i].max())
     np.save('tuning.npy', Z)
else:
    Z = np.load('tuning.npy')

fig = plt.figure(figsize=(15,5), facecolor='w')
ax = plt.subplot(1,1,1)
ax.tick_params(direction="outward")
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
# ax.spines['left'].set_color('none')
ax.spines['bottom'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')
# ax.spines['bottom'].set_position(('data',-0.05))

for i in [3,5,10,15]:
    x,y = polar_to_logpolar( i/90.0, 0 )
    index = int(x*Z.shape[1])
    plt.plot(X,Z[:,index], linewidth=1.5, color='k')

plt.xlim(0.0, 25.0)
plt.ylim(0.0,  1.1)
plt.yticks([0.0,1.0],['0','500'])
#plt.vlines([3,5,10,15], [0,0,0,0], [1.1,1.1,1.1,1.1],  linewidth=1, color='.75')
plt.xticks([0,3,5,10,15,25])
plt.xlabel('Target eccentricity ($^\circ$)')
plt.ylabel('Discharge rate (spike/s)')
plt.savefig('fig-tuning-curves.pdf')
plt.show()
