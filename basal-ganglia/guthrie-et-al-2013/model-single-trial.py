#!/usr/bin/env python
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier, Meropi Topalidou
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
#               Meropi Topalidou (Meropi.Topalidou@inria.fr)
# -----------------------------------------------------------------------------
# References:
#
# * Interaction between cognitive and motor cortico-basal ganglia loops during
#   decision making: a computational study. M. Guthrie, A. Leblois, A. Garenne,
#   and T. Boraud. Journal of Neurophysiology, 109:3025–3040, 2013.
#
# * A long Journey into Reproducible Computational Neurosciences (submitted)
#   Meropi Topalidou, Arthur Leblois, Thomas Boraud, Nicolas P. Rougier
# -----------------------------------------------------------------------------
from dana import *
import matplotlib.pyplot as plt

# Parameters
# -----------------------------------------------------------------------------
# Population size
n = 4

# Default trial duration
duration = 3.0*second

# Default Time resolution
dt = 1.0*millisecond

# Initialization of the random generator (reproductibility !)
np.random.seed(1)

# Sigmoid parameter
Vmin       =  0.0
Vmax       = 20.0
Vh         = 16.0
Vc         =  3.0

# Thresholds
Cortex_h   =  -3.0
Striatum_h =   0.0
STN_h      = -10.0
GPi_h      =  10.0
Thalamus_h = -40.0

# Time constants
Cortex_tau   = 0.01
Striatum_tau = 0.01
STN_tau      = 0.01
GPi_tau      = 0.01
Thalamus_tau = 0.01

# Noise level (%)
Cortex_N   =   0.01
Striatum_N =   0.001
STN_N      =   0.001
GPi_N      =   0.03
Thalamus_N =   0.001


# Helper functions
# -----------------------------------------------------------------------------
def sigmoid(V,Vmin=Vmin,Vmax=Vmax,Vh=Vh,Vc=Vc):
    return  Vmin + (Vmax-Vmin)/(1.0+np.exp((Vh-V)/Vc))

def noise(Z, level):
    Z = (1+np.random.uniform(-level/2,level/2,Z.shape))*Z
    return np.maximum(Z,0.0)

def init_weights(L, gain=1):
    Wmin, Wmax = 0.25, 0.75
    W = L._weights
    N = np.random.normal(0.5, 0.005, W.shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    L._weights = gain*W*(Wmin + (Wmax - Wmin)*N)


# Populations
# -----------------------------------------------------------------------------
Cortex_cog   = zeros((n,1), """dV/dt = (-V + I + Iext - Cortex_h)/Cortex_tau;
                               U = noise(V,Cortex_N); I; Iext""")
Cortex_mot   = zeros((1,n), """dV/dt = (-V + I + Iext - Cortex_h)/Cortex_tau;
                               U = noise(V,Cortex_N); I; Iext""")
Cortex_ass   = zeros((n,n), """dV/dt = (-V + I + Iext - Cortex_h)/Cortex_tau;
                               U = noise(V,Cortex_N); I; Iext""")
Striatum_cog = zeros((n,1), """dV/dt = (-V + I - Striatum_h)/Striatum_tau;
                               U = noise(sigmoid(V), Striatum_N); I""")
Striatum_mot = zeros((1,n), """dV/dt = (-V + I - Striatum_h)/Striatum_tau;
                               U = noise(sigmoid(V), Striatum_N); I""")
Striatum_ass = zeros((n,n), """dV/dt = (-V + I - Striatum_h)/Striatum_tau;
                               U = noise(sigmoid(V), Striatum_N); I""")
STN_cog      = zeros((n,1), """dV/dt = (-V + I - STN_h)/STN_tau;
                               U = noise(V,STN_N); I""")
STN_mot      = zeros((1,n), """dV/dt = (-V + I - STN_h)/STN_tau;
                               U = noise(V,STN_N); I""")
GPi_cog      = zeros((n,1), """dV/dt = (-V + I - GPi_h)/GPi_tau;
                               U = noise(V,GPi_N); I""")
GPi_mot      = zeros((1,n), """dV/dt = (-V + I - GPi_h)/GPi_tau;
                               U = noise(V,GPi_N); I""")
Thalamus_cog = zeros((n,1), """dV/dt = (-V + I - Thalamus_h)/Thalamus_tau;
                               U = noise(V,Thalamus_N); I""")
Thalamus_mot = zeros((1,n), """dV/dt = (-V + I - Thalamus_h)/Thalamus_tau;
                               U = noise(V, Thalamus_N); I""")


# Connectivity
# -----------------------------------------------------------------------------
L = DenseConnection( Cortex_cog('U'),   Striatum_cog('I'), 1.0)
init_weights(L)
L = DenseConnection( Cortex_mot('U'),   Striatum_mot('I'), 1.0)
init_weights(L)
L = DenseConnection( Cortex_ass('U'),   Striatum_ass('I'), 1.0)
init_weights(L)
L = DenseConnection( Cortex_cog('U'),   Striatum_ass('I'), np.ones((1,2*n+1)))
init_weights(L,0.2)
L = DenseConnection( Cortex_mot('U'),   Striatum_ass('I'), np.ones((2*n+1,1)))
init_weights(L,0.2)

DenseConnection( Cortex_cog('U'),   STN_cog('I'),       1.0 )
DenseConnection( Cortex_mot('U'),   STN_mot('I'),       1.0 )
DenseConnection( Striatum_cog('U'), GPi_cog('I'),      -2.0 )
DenseConnection( Striatum_mot('U'), GPi_mot('I'),      -2.0 )
DenseConnection( Striatum_ass('U'), GPi_cog('I'),      -2.0*np.ones((1,2*n+1)))
DenseConnection( Striatum_ass('U'), GPi_mot('I'),      -2.0*np.ones((2*n+1,1)))
DenseConnection( STN_cog('U'),      GPi_cog('I'),       1.0*np.ones((2*n+1,1)) )
DenseConnection( STN_mot('U'),      GPi_mot('I'),       1.0*np.ones((1,2*n+1)) )
DenseConnection( GPi_cog('U'),      Thalamus_cog('I'), -0.5 )
DenseConnection( GPi_mot('U'),      Thalamus_mot('I'), -0.5 )
DenseConnection( Thalamus_cog('U'), Cortex_cog('I'),    1.0 )
DenseConnection( Thalamus_mot('U'), Cortex_mot('I'),    1.0 )
DenseConnection( Cortex_cog('U'),   Thalamus_cog('I'),  0.4 )
DenseConnection( Cortex_mot('U'),   Thalamus_mot('I'),  0.4 )


# Trial setup
# -----------------------------------------------------------------------------
@clock.at(500*millisecond)
def set_trial(t):
    m1,m2 = np.random.randint(0,4,2)
    while m2 == m1:
        m2 = np.random.randint(4)
    c1,c2 = np.random.randint(0,4,2)
    while c2 == c1:
        c2 = np.random.randint(4)
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0
    v = 7
    Cortex_mot['Iext'][0,m1]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_mot['Iext'][0,m2]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_cog['Iext'][c1,0]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_cog['Iext'][c2,0]  = v + np.random.normal(0,v*Cortex_N)
    Cortex_ass['Iext'][c1,m1] = v + np.random.normal(0,v*Cortex_N)
    Cortex_ass['Iext'][c2,m2] = v + np.random.normal(0,v*Cortex_N)

@clock.at(2500*millisecond)
def set_trial(t):
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0


# Measurements
# -----------------------------------------------------------------------------
dtype = [ ("cortex",   [("mot", float, 4),
                        ("cog", float, 4),
                        ("ass", float, 16)]),
          ("striatum", [("mot", float, 4),
                        ("cog", float, 4),
                        ("ass", float, 16)]),
          ("GPi",      [("mot", float, 4),
                        ("cog", float, 4)]),
          ("thalamus", [("mot", float, 4),
                        ("cog", float, 4)]),
          ("STN",      [("mot", float, 4),
                        ("cog", float, 4)])]
n = int(duration/dt)+1
timesteps = np.zeros(n)
records = np.zeros(n, dtype=dtype)
record_index = 0

@after(clock.tick)
def register(t):
    global record_index

    i = record_index
    timesteps[i] = t
    records["cortex"]["mot"][i] = Cortex_mot['U'].ravel()
    records["cortex"]["cog"][i] = Cortex_cog['U'].ravel()
    records["cortex"]["ass"][i] = Cortex_ass['U'].ravel()
    records["striatum"]["mot"][i] = Striatum_mot['U'].ravel()
    records["striatum"]["cog"][i] = Striatum_cog['U'].ravel()
    records["striatum"]["ass"][i] = Striatum_ass['U'].ravel()
    records["STN"]["mot"][i] = STN_mot['U'].ravel()
    records["STN"]["cog"][i] = STN_cog['U'].ravel()
    records["GPi"]["mot"][i] = GPi_mot['U'].ravel()
    records["GPi"]["cog"][i] = GPi_cog['U'].ravel()
    records["thalamus"]["mot"][i] = Thalamus_mot['U'].ravel()
    records["thalamus"]["cog"][i] = Thalamus_cog['U'].ravel()
    record_index += 1


# Simulation
# -----------------------------------------------------------------------------
run(time=duration, dt=dt)


# Display 1
# -----------------------------------------------------------------------------
if 0:
    fig = plt.figure(figsize=(12,5))
    plt.subplots_adjust(bottom=0.15)

    fig.patch.set_facecolor('.9')
    ax = plt.subplot(1,1,1)

    plt.plot(timesteps, records["cortex"]["cog"][:,0],c='r', label="Cognitive Cortex")
    plt.plot(timesteps, records["cortex"]["cog"][:,1],c='r')
    plt.plot(timesteps, records["cortex"]["cog"][:,2],c='r')
    plt.plot(timesteps, records["cortex"]["cog"][:,3],c='r')
    plt.plot(timesteps, records["cortex"]["mot"][:,0],c='b', label="Motor Cortex")
    plt.plot(timesteps, records["cortex"]["mot"][:,1],c='b')
    plt.plot(timesteps, records["cortex"]["mot"][:,2],c='b')
    plt.plot(timesteps, records["cortex"]["mot"][:,3],c='b')

    plt.xlabel("Time (seconds)")
    plt.ylabel("Activity (Hz)")
    plt.legend(frameon=False, loc='upper left')
    plt.xlim(0.0,duration)
    plt.ylim(0.0,60.0)

    plt.xticks([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0],
               ['0.0','0.5\n(Trial start)','1.0','1.5', '2.0','2.5\n(Trial stop)','3.0'])
    # plt.savefig("model-single-trial.pdf")
    plt.show()


# Display 2
# -----------------------------------------------------------------------------
if 0:
    fig = plt.figure(figsize=(18,12))
    fig.patch.set_facecolor('1.0')

    def subplot(rows,cols,n, alpha=0.0):
        ax = plt.subplot(rows,cols,n)
        ax.patch.set_facecolor("k")
        ax.patch.set_alpha(alpha)

        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.spines['bottom'].set_color('none')
        ax.yaxis.set_ticks_position('left')
        ax.yaxis.set_tick_params(direction="outward")
        return ax

    ax = subplot(5,3,1)
    ax.set_title("MOTOR", fontsize=24)
    ax.set_ylabel("STN", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, records["STN"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,2)
    ax.set_title("COGNITIVE", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, records["STN"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,3,alpha=0)
    ax.set_title("ASSOCIATIVE", fontsize=24)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['left'].set_color('none')


    ax = subplot(5,3,4)
    ax.set_ylabel("CORTEX", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, records["cortex"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,5)
    for i in range(4):
        plt.plot(timesteps, records["cortex"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,6)
    for i in range(16):
        plt.plot(timesteps, records["cortex"]["ass"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,7)
    ax.set_ylabel("STRIATUM", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, records["striatum"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,8)
    for i in range(4):
        plt.plot(timesteps, records["striatum"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,9)
    for i in range(16):
        plt.plot(timesteps, records["striatum"]["ass"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,10)
    ax.set_ylabel("GPi", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, records["GPi"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,11)
    for i in range(4):
        plt.plot(timesteps, records["GPi"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,13)
    ax.set_ylabel("THALAMUS", fontsize=24)
    for i in range(4):
        plt.plot(timesteps, records["thalamus"]["mot"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    ax = subplot(5,3,14)
    for i in range(4):
        plt.plot(timesteps, records["thalamus"]["cog"][:,i], c='k', lw=.5)
    ax.set_xticks([])

    # plt.savefig("model-single-trial-all.pdf")
    plt.show()
