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
# np.random.seed(1)

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

# Learning parameters
decision_threshold = 40
alpha_c     = 0.05
alpha_LTP  = 0.002
alpha_LTD  = 0.001
Wmin, Wmax = 0.25, 0.75


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

def reset():
    for group in network.__default_network__._groups:
        group['U'] = 0
        group['V'] = 0
        group['I'] = 0
    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0

def clip(V, Vmin, Vmax):
    return np.minimum(np.maximum(V, Vmin), Vmax)


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

cues_mot = np.array([0,1,2,3])
cues_cog = np.array([0,1,2,3])
cues_value = np.ones(4) * 0.5
cues_reward = np.array([3.0,2.0,1.0,0.0])/3.0


# Connectivity
# -----------------------------------------------------------------------------
W = DenseConnection( Cortex_cog('U'),   Striatum_cog('I'), 1.0)
init_weights(W)
W_cortex_cog_to_striatum_cog = W

W = DenseConnection( Cortex_mot('U'),   Striatum_mot('I'), 1.0)
init_weights(W)
W = DenseConnection( Cortex_ass('U'),   Striatum_ass('I'), 1.0)
init_weights(W)
W = DenseConnection( Cortex_cog('U'),   Striatum_ass('I'), np.ones((1,2*n+1)))
init_weights(W,0.2)
W = DenseConnection( Cortex_mot('U'),   Striatum_ass('I'), np.ones((2*n+1,1)))
init_weights(W,0.2)
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

    np.random.shuffle(cues_cog)
    np.random.shuffle(cues_mot)
    c1,c2 = cues_cog[:2]
    m1,m2 = cues_mot[:2]

    v = 7

    Cortex_mot['Iext'] = 0
    Cortex_cog['Iext'] = 0
    Cortex_ass['Iext'] = 0
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


# Learning
# -----------------------------------------------------------------------------
P, R = [], []

@after(clock.tick)
def register(t):
    U = np.sort(Cortex_mot["V"]).ravel()

    # No motor decision yet
    if abs(U[-1] - U[-2]) < decision_threshold: return

    # A motor decision has been made
    c1, c2 = cues_cog[:2]
    m1, m2 = cues_mot[:2]
    mot_choice = np.argmax(Cortex_mot['V'])
    cog_choice = np.argmax(Cortex_cog['V'])

    # The actual cognitive choice may differ from the cognitive choice
    # Only the motor decision can designate the chosen cue
    if mot_choice == m1:
        choice = c1
    else:
        choice = c2

    if choice == min(c1,c2):
        P.append(1)
    else:
        P.append(0)

    # Compute reward
    reward = np.random.uniform(0,1) < cues_reward[choice]
    R.append(reward)

    # Compute prediction error
    #error = cues_reward[choice] - cues_value[choice]
    error = reward - cues_value[choice]

    # Update cues values
    cues_value[choice] += error* alpha_c

    # Learn
    lrate = alpha_LTP if error > 0 else alpha_LTD
    dw = error * lrate * Striatum_cog['U'][choice][0]

    W = W_cortex_cog_to_striatum_cog
    w = clip(W.weights[choice, choice] + dw, Wmin, Wmax)
    W.weights[choice,choice] = w

    if choice == min(c1,c2):
        print "Choice (%d/%d) : %d (best)" % (c1,c2,choice)
    else:
        print "Choice (%d/%d) : %d (bad)" % (c1,c2,choice)
    print "Reward (%.2f%%) : %d" % (cues_reward[choice],reward)
    print "Mean performance: ", np.array(P).mean()
    print "Mean reward:      ", np.array(R).mean()
    print

    # Start a new trial
    reset()
    clock.reset()


# Simulation
# -----------------------------------------------------------------------------
for i in range(1200):
    print "Trial", i
    run(time=duration, dt=dt)
