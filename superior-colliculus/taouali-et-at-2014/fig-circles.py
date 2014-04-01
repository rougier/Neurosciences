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
    radius = 0.01

    polar_frame(ax1, legend=True)
    def draw(X,Y):
        poly = Polygon(zip(X,Y), facecolor='b', edgecolor='none', alpha=0.1)
        ax1.add_patch(poly)
        ax1.plot(X,Y, c='k', lw=.5, alpha=.5)

    rho = 30/90.0
    theta  = 0
    for radius in np.linspace(.005,.3,10):
        x0,y0 = polar_to_cartesian(rho,theta)
        T = np.linspace(0,2*np.pi,1000)
        X = x0 + radius*np.cos(T)
        Y = y0 + radius*np.sin(T)
        draw(X,Y)

    logpolar_frame(ax2, legend=True)
    def draw(X,Y):
        R,T = cartesian_to_polar(X,Y)
        X,Y = polar_to_logpolar(R,T)
        X = 2*X
        Y = 2*Y-1
        poly = Polygon(zip(X,Y), facecolor='r', edgecolor='none', alpha=0.1)
        ax2.add_patch(poly)
        ax2.plot(X,Y, c='k', lw=.5, alpha=.5)

    rho = 30/90.0
    theta  = 0
    for radius in np.linspace(.005,.3,10):
        x0,y0 = polar_to_cartesian(rho,theta)
        T = np.linspace(0,2*np.pi,1000)
        X = x0 + radius*np.cos(T)
        Y = y0 + radius*np.sin(T)
        draw(X,Y)

    plt.savefig("fig-circles.pdf")
    plt.show()
