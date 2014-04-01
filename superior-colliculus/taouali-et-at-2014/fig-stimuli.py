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
import os
import numpy as np

from helper import *
from graphics import *
from projections import *


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.axes_grid1 import ImageGrid
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    P = retina_to_colliculus( (4*1024,4*512), (512,512) )


    fig = plt.figure(figsize=(10,8), facecolor='w')
    ax1, ax2 = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.5)

    def stimulus(shape, position, size):
        x,y = polar_to_cartesian(position[0]/90.0, np.pi*position[1]/180.0)
        Y,X = np.mgrid[0:shape[0],0:shape[1]]
        X = X/float(shape[1])
        Y = 2*Y/float(shape[0])-1
        r = (X-x)**2+(Y-y)**2
        return np.exp(-0.5*r/(size/90.0))

    polar_frame(ax1, legend=True)
    zax = zoomed_inset_axes(ax1, 6, loc=1)
    polar_frame(zax, zoom=True)
    zax.set_xlim(0.0, 0.1)
    zax.set_xticks([])
    zax.set_ylim(-.05, .05)
    zax.set_yticks([])
    zax.set_frame_on(True)
    mark_inset(ax1, zax, loc1=2, loc2=4, fc="none", ec="0.5")

    R  = stimulus((4*1024,4*512), (1,0), 0.005)
    R += stimulus((4*1024,4*512), (5,0), 0.005)
    polar_imshow(ax1, R)
    polar_imshow(zax, R)

    logpolar_frame(ax2, legend=True)
    SC = R[P[...,0], P[...,1]]
    logpolar_imshow(ax2, SC)

    plt.savefig("fig-stimuli.pdf")
    plt.show()
