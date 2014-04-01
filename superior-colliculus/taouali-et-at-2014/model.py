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
import matplotlib.pyplot as plt
from numpy.fft import rfft2, irfft2
from numpy.fft import fftshift, ifftshift
from scipy.ndimage.interpolation import zoom

from helper import *
from stimulus import *
from graphics import *
from parameters import *
from projections import *

n = colliculus_shape[0]
K = A_e*gaussian((2*n+1,2*n+1), sigma_e) - A_i #*gaussian((2*n+1,2*n+1), sigma_i)

# Prepare fft
K_shape = np.array(K.shape)
fft_shape = np.array(best_fft_shape(colliculus_shape+K_shape//2))
K_fft = rfft2(K,fft_shape)
i0,j0 = K.shape[0]//2, K.shape[1]//2
i1,j1 = i0+colliculus_shape[0], j0+colliculus_shape[1]

P = retina_projection()
# R = np.maximum( stimulus((5.0,-25.0)), stimulus((5.0,25.0)) )
R = stimulus((25.0,0.0))

SCu = np.zeros(colliculus_shape)
SCv = np.zeros(colliculus_shape)
I_high = R[P[...,0], P[...,1]]
I = zoom(I_high, colliculus_shape/projection_shape)
I += np.random.uniform(-noise/2,+noise/2,I.shape)

s = fft_shape
n = int(duration/dt)

for i in range(int(duration/dt)):
    L = (irfft2(rfft2(SCv,s)*K_fft,s)).real[i0:i1,j0:j1]
    SCu += dt/tau*(-SCu + (scale*L + I)/alpha)
    SCv = np.minimum(np.maximum(0,SCu),1)
    # SCv = np.maximum(0,SCu)

fig = plt.figure(figsize=(10,8), facecolor='w')
ax1, ax2 = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.5)
polar_frame(ax1, legend=True)
zax = zoomed_inset_axes(ax1, 6, loc=1)
polar_frame(zax, zoom=True)
zax.set_xlim(0.0, 0.1)
zax.set_xticks([])
zax.set_ylim(-.05, .05)
zax.set_yticks([])
zax.set_frame_on(True)
mark_inset(ax1, zax, loc1=2, loc2=4, fc="none", ec="0.5")
polar_imshow(ax1, R)
polar_imshow(zax, R)
logpolar_frame(ax2, legend=True)
logpolar_imshow(ax2, SCv)

plt.show()
