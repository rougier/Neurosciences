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


class Model:

    def __init__(self):
        # Retina
        self.R = np.zeros(retina_shape)

        # Superior colliculus
        self.SC_V = np.zeros(colliculus_shape)
        self.SC_U = np.zeros(colliculus_shape)

        # Projection from retina to colliculus
        self.P = retina_projection()

        # Retina mask
        D = disc((retina_shape[0],retina_shape[0]),
                 (retina_shape[0]//2,retina_shape[0]//2),
                 retina_shape[0]//2)
        self.R_mask = D[:,retina_shape[1]:]

        # Colliculus mask
        self.SC_mask = self.R_mask[self.P[...,0], self.P[...,1]]
        self.SC_mask = zoom(self.SC_mask, colliculus_shape/projection_shape)

        # Parameters
        self.sigma_e  = sigma_e
        self.A_e      = A_e
        self.sigma_i  = sigma_i
        self.A_i      = A_i
        self.alpha    = alpha
        self.tau      = tau
        self.scale    = scale
        self.noise    = noise
        self.lesion   = None

        # Lateral weights
        # DoG
        # K = A_e*gaussian((2*n+1,2*n+1), sigma_e) - A_i*gaussian((2*n+1,2*n+1), sigma_i)
        # Constant inhibition
        K = A_e*gaussian((2*n+1,2*n+1), sigma_e) - A_i

        # FFT for lateral weights
        K_shape = np.array(K.shape)
        self.fft_shape = np.array(best_fft_shape(colliculus_shape+K_shape//2))
        self.K_fft = rfft2(K,self.fft_shape)
        i0,j0 = K.shape[0]//2, K.shape[1]//2
        i1,j1 = i0+colliculus_shape[0], j0+colliculus_shape[1]
        self.K_indices = i0,i1,j0,j1

    def sigmoid(self, x):
        ''' Sigmoid function of the form 1/(1+exp(-x)) '''
        #return 1.0/(1.0+np.exp(2-4*x))
        return 1.0/(1.0+np.exp(1-5*x))

    def reset(self):
        self.R[...] = 0
        self.SC_U[...] = 0
        self.SC_V[...] = 0

    def make_lesion(self, lesion = None):
        if lesion is not None:
            position,size = lesion
            rho,theta = position
            rho,theta = rho/90.0, np.pi*theta/180.0
            x,y = polar_to_logpolar(rho,theta)
            x = int(x*colliculus_shape[0])
            y = int(y*colliculus_shape[1])
            radius = ((0.5*size/90.0)*colliculus_shape[0])
            self.lesion = disc(colliculus_shape, center = (y,x), radius=radius)
        else:
            self.lesion = None


    def run(self, duration=duration, dt=dt, epsilon=0.01):
        # Set some input
        # R = np.maximum( stimulus((5.0,-25.0)), stimulus((5.0,25.0)) )
        # R = stimulus((15.0,0.0))

        # Project retina to input
        I_high = self.R[self.P[...,0], self.P[...,1]]
        I = zoom(I_high, colliculus_shape/projection_shape)

        # White noise in the input
        # I += np.random.uniform(-noise/2,+noise/2,I.shape)
        self.I = I * (1+np.random.normal(0, 5*noise, I.shape))

        s = self.fft_shape
        i0,i1,j0,j1 = self.K_indices

        Z = np.zeros((int(duration/dt), self.SC_V.shape[0], self.SC_V.shape[1]))

        for i in range( int(duration/dt) ):

            Z[i] = self.SC_V

            L = (irfft2(rfft2(self.SC_V,s)*self.K_fft, s)).real[i0:i1,j0:j1]
            dU = dt/self.tau*(-self.SC_U + (self.scale*L + I)/self.alpha)
            self.SC_U += dU

            if self.lesion is not None:
                self.SC_U *= (1-self.lesion)
            self.SC_V = np.minimum(np.maximum(0,self.SC_U),1)
            if np.abs(dU).sum() < epsilon:
                break

            # Correlated Gaussian noise
            self.SC_V = self.SC_V * (1+np.random.normal(0,noise,self.SC_V.shape))
            # self.SC_V *= self.SC_mask


        self.SC_V *= self.SC_mask
        self.SC_U *= self.SC_mask

        return Z
