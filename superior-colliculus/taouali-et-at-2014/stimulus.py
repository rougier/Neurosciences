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

from parameters import *
from projections import *


def stimulus(position,
             size = stimulus_size,
             intensity = stimulus_intensity,
             shape = retina_shape):
    """
    Parameters
    ----------

    position : (float,float)
        Position in degrees

    size : float
        Size in degrees

    intensity: float
        Intensity
    """

    x,y = polar_to_cartesian(position[0]/90.0, np.pi*position[1]/180.0)
    x,y = x*90, y*90

    Y,X = np.mgrid[0:shape[0],0:shape[1]]
    X = (  X/float(shape[1])  ) * 90
    Y = (2*Y/float(shape[0])-1) * 90

    R = ((X-x)**2+(Y-y)**2)

    # For a gaussian of type:
    # f(x) = a * exp( - (x-b)**2 / (2*c*c)) + d
    #
    # Full width at half maximum (FWHM) = 2sqrt(2ln2)*c = 2.35482 * c
    # -> FWHM = size = 2.35482 * c
    #
    c = size/2.35482

    return intensity * np.exp(-R/(2*c*c))
