#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright INRIA
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#
# DANA is a computing framework for the simulation of distributed,
# asynchronous, numerical and adaptive models.
#
# This software is governed by the CeCILL license under French law and abiding
# by the rules of distribution of free software. You can use, modify and/ or
# redistribute the software under the terms of the CeCILL license as circulated
# by CEA, CNRS and INRIA at the following URL: http://www.cecill.info.
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
#
# References:
# -----------
#
# * Girard B, Tabareau N, Pham QC, Berthoz A, Slotine JJ (2008) "Where
#   neuroscience and dynamic system theory meet autonomous robotics: a
#   contracting basal ganglia model for action selection". Neural Networks
#   21:628-41
#
# -----------------------------------------------------------------------------
from dana import *


figs = {}

def draw_group(name, data, title=''):
    ''' '''
    axes = plt.gca()
    x = np.arange(len(data))
    axes.grid()
    rects = axes.bar(left=x,height=np.ones_like(data),
                     width=.75, align='center', alpha=0.5)
    axes.set_ylim([0,1])
    plt.xticks(x)
    axes.set_title(title, fontsize=10)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(10)
    figs[name] = axes,rects

def update_group(name, data):
    ''' '''
    axes,rects = figs[name]
    for rect,h in zip(rects,data):
        rect.set_height(h)

def draw_plot(name, title=''):
    ''' '''
    axes = plt.gca()
    axes.grid()
    line, = axes.plot([1],[1])
    axes.set_xlim([0,10])
    axes.set_ylim([-0.1,+0.7])
    axes.set_title(title, fontsize=10)
    for label in axes.get_xticklabels() + axes.get_yticklabels():
        label.set_fontsize(10)
    plt.xlabel('Time (s)', fontsize=10)
    figs[name] = axes,line

def update_plot(name, x, y):
    ''' '''
    axes, line = figs[name]
    line.set_xdata(x)
    line.set_ydata(y)

# dopamine level
DA_1 = 0.2   
DA_2 = 0.2

# Number of channels
n = 6

# Time constant
tau  =  0.02

# Resting levels
eSt1 = +0.10
eSt2 = +0.10
eSTN = -0.5
eGPe = -0.10
eGPi = -0.10
eVLT = -0.1
ePFC = +0.00
eTRN = +0.00

# Connections strength
STN_GPe = +0.7
St2_GPe = -0.4
St1_GPe = -0.4

STN_GPi = +0.7
GPe_GPi = -0.08
St1_GPi = -0.4

GPe_STN = -0.45
PFC_STN = +0.58
PC_STN  = +0.0

PFC_St1 = +0.1
PC_St1  = +0.9
GPe_St1 = -1.0

PFC_St2 = +0.1
PC_St2  = +0.9
GPe_St2 = -1.0

GPi_VLT = -0.18
PFC_VLT = +0.6
TRN_VLT = -0.35

PFC_TRN = +0.35
GPi_TRN = -0.0
VLT_TRN = +0.35

VLT_PFC = +0.6


# Posterior Cortex
PC  = zeros(n, '''V''')

# Striatum D1: medium spiny neurons of the striatum with D1 dopamine receptors
St1 = zeros(n, '''dU/dt = 1/tau*(-U + PC_ + PFC_ + GPe_-eSt1 )
                   V    = np.minimum(np.maximum(U,0),1)
                   PC_; PFC_;GPe_''')

# Striatum D2: medium spiny neurons of the striatum with D2 dopamine receptors
St2 = zeros(n, '''dU/dt = 1/tau*(-U + PC_ + PFC_ + GPe_-eSt2)
                   V    = np.minimum(np.maximum(U,0),1)
                   PC_; PFC_;GPe_''')

# Sub-Thalamic Nucleus 
STN = zeros(n, '''dU/dt = 1/0.05*(-U + PC_ + PFC_ + GPe_-eSTN)
                   V    = np.minimum(np.maximum(U,0),1)
                   PC_; PFC_; GPe_''')

# External Globus Pallidus
GPe = zeros(n, '''dU/dt = 1/tau*(-U + STN_ + St2_ +St1_-eGPe)
                   V    = np.minimum(np.maximum(U,0),1)
                   STN_; St2_ ;St1_''')

# External Globus Pallidus
GPi = zeros(n, '''dU/dt = 1/tau*(-U + STN_ + St1_ + GPe_-eGPi)
                   V    = np.minimum(np.maximum(U,0),1)
                   STN_; St1_; GPe_''')

#  Ventro-Lateral Thalamus
VLT = zeros(n, '''dU/dt = 1/0.05*(-U + PFC_ + TRN_ + GPi_-eVLT)
                   V    = np.minimum(np.maximum(U,0),2)
                   PFC_; TRN_; GPi_''')

# Prefrontal Cortex
PFC = zeros(n, '''dU/dt = 1/(tau*4)*(-U + PC_ + VLT_-ePFC)
                   V    = np.minimum(np.maximum(U,0),2)
                   PC_; VLT_''')

# Thalamic Reticular Nucleus
TRN = zeros(1, '''dU/dt = 1/0.05*(-U + PFC_ + VLT_ + GPi_-eTRN)
                   V    = np.minimum(np.maximum(U,0),2)
                   PFC_; VLT_; GPi_''')

# St1 connections
SparseConnection( PC('V'),  St1('PC_'),   (1 + DA_1 )*PC_St1 * np.ones(1) )
SparseConnection( PFC('V'), St1('PFC_'), (1 + DA_1 )*PFC_St1 * np.ones(1) )
SparseConnection( GPe('V'), St1('GPe_'),   (1 + DA_1 )*GPe_St1 * np.ones(1) )

# St2 connections
SparseConnection( PC('V'),  St2('PC_'),   (1 - DA_2)*PC_St2 * np.ones(1) )
SparseConnection( PFC('V'), St2('PFC_'), (1 - DA_2 )*PFC_St2 * np.ones(1) )
SparseConnection( GPe('V'),  St2('GPe_'),   (1 - DA_2 )*GPe_St2 * np.ones(1) )

# STN connections
SparseConnection( PC('V') , STN('PC_'),   PC_STN * np.ones(1) )
SparseConnection( PFC('V'), STN('PFC_'), PFC_STN * np.ones(1) )
SparseConnection( GPe('V'), STN('GPe_'), GPe_STN * np.ones((n,n)) )

# GPe connections
DenseConnection(  STN('V'), GPe('STN_'), STN_GPe * np.ones((n,n)) )
SparseConnection( St2('V'), GPe('St2_'), St2_GPe * np.ones(1) )
SparseConnection( St1('V'), GPe('St1_'), St1_GPe * np.ones(1) )

# GPi connections
DenseConnection(  STN('V'), GPi('STN_'), STN_GPi * np.ones((n,n)) )
SparseConnection( St1('V'), GPi('St1_'), St1_GPi * np.ones(1) ) 
SparseConnection( GPe('V'), GPi('GPe_'), GPe_GPi * np.ones((n,n)) ) 

# VLT connections
SparseConnection( PFC('V'), VLT('PFC_'), PFC_VLT * np.ones(1) )
SparseConnection( GPi('V'), VLT('GPi_'), GPi_VLT * np.ones(1) ) 
K = TRN_VLT * np.ones((n,))
DenseConnection(  TRN('V'), VLT('TRN_'), K )

# PFC connections
SparseConnection( VLT('V'), PFC('VLT_'), VLT_PFC * np.ones(1) )
SparseConnection(  PC('V'), PFC('PC_'),            np.ones(1) )

# TRN connections
DenseConnection( PFC('V'), TRN('PFC_'), PFC_TRN * np.ones((1,n)) ) 
DenseConnection( VLT('V'), TRN('VLT_'), VLT_TRN * np.ones((1,n)) ) 
SparseConnection( GPi('V'), TRN('GPi_'), GPi_TRN * np.ones(1) )


# Draw figures
plt.ion()
fig = plt.figure(figsize=(15,12))
plt.subplot(4,3,1)
draw_group('PC',  PC['V'],  'Posterior Cortex (PC)')
plt.subplot(4,3,2)
draw_group('PFC', PFC['V'], 'Prefrontal Cortex (PFC)')
plt.subplot(4,3,3)
draw_group('STN', STN['V'], 'Sub-Thalamic Nucleus (STN)')
plt.subplot(4,3,4)
draw_group('St1', St1['V'], 'Striatum D1 (St1)')
plt.subplot(4,3,5)
draw_group('St2', St2['V'], 'Striatum D2 (St2)')
plt.subplot(4,3,6)
draw_group('GPe', GPe['V'], 'External globus pallidus (GPe)')
plt.subplot(4,3,7)
draw_group('GPi', GPi['V'], 'Internal globus pallidus (GPi)')
plt.subplot(4,3,8)
draw_group('VLT', VLT['V'], 'Ventro Lateral Thalamus (VLT)')
plt.subplot(4,3,9)
draw_group('TRN', TRN['V'], 'Thalamic Reticular Nucleus (TRN)')
plt.subplot(4,3,10)
draw_plot('GPi_1', 'GPi channel 1')
plt.subplot(4,3,11)
draw_plot('GPi_2', 'GPi channel 2')
plt.subplot(4,3,12)
draw_plot('GPi_3', 'GPi channel 3-6')

GPi_1 = []
GPi_2 = []
GPi_3 = []



@clock.at(2*second)
def update_PC(t):
    PC.V[...][0] = .4

@clock.at(4*second)
def update_PC(t):
    PC.V[...][1] = .6

@clock.at(6*second)
def update_PC(t):
    PC.V[...][0] = .6

@clock.at(8*second)
def update_PC(t):
    PC.V[...][0] = .4

# Refresh figures every 10 miliseconds
@clock.every(25*millisecond)
def update_figure(t):

    GPi_1.append(GPi['V'][0])
    GPi_2.append(GPi['V'][1])
    GPi_3.append(GPi['V'][2])

    update_group('PC',  PC['V'])
    update_group('PFC', PFC['V'])
    update_group('STN', STN['V'])
    update_group('St1', St1['V'])
    update_group('St2', St2['V'])
    update_group('GPe', GPe['V'])
    update_group('GPi', GPi['V'])
    update_group('VLT', VLT['V'])
    update_group('TRN', TRN['V'])

    update_plot('GPi_1', np.arange(len(GPi_1))*25*millisecond, GPi_1)
    update_plot('GPi_2', np.arange(len(GPi_2))*25*millisecond, GPi_2)
    update_plot('GPi_3', np.arange(len(GPi_3))*25*millisecond, GPi_3)

    plt.draw()

# Run simulation for 1 second
run(time=10*second, dt=1*millisecond)
plt.ioff()
plt.savefig('Girard_et_al_2008.pdf')
plt.show()
