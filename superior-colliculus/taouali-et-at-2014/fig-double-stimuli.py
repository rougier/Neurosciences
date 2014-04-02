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
from model import *


def plot(model, filename, zoom=False):

    fig = plt.figure(figsize=(10,8), facecolor='w')
    ax1, ax2 = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.5)
    polar_frame(ax1, legend=True)
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

    logpolar_frame(ax2, legend=True)
    logpolar_imshow(ax2, model.SC_V)
    plt.savefig(filename)
    plt.show()


model = Model()

# (2,0) and (5,0)
model.reset()
model.R = np.maximum( stimulus((2.0, 0.0)), stimulus((5.0, 0.0)) )
model.run(duration = 5*second)
plot(model, "double-stimuli-(2,0)-(5,0).pdf", zoom=True)

# (5,-10) and (5,+10)
model.reset()
model.R = np.maximum( stimulus((5.0, +10.0)), stimulus((5.0, -10.0)) )
model.run(duration = 5*second)
plot(model, "double-stimuli-(5,+10)-(5,-10).pdf", zoom=True)

# (5,-25) and (5,+25)
model.reset()
model.R = np.maximum( stimulus((5.0, +25.0)), stimulus((5.0, -25.0)) )
model.run(duration = 5*second)
plot(model, "double-stimuli-(5,+25)-(5,-25).pdf", zoom=True)
