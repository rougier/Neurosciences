# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2015, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
from c_dana import *
from parameters import *

clamp   = Clamp(min=0, max=1000)
sigmoid = Sigmoid(Vmin=Vmin, Vmax=Vmax, Vh=Vh, Vc=Vc)

CTX = AssociativeStructure(
                 tau=tau, rest=CTX_rest, noise=noise, activation=clamp )
STR = AssociativeStructure(
                 tau=tau, rest=STR_rest, noise=noise, activation=sigmoid )
STN = Structure( tau=tau, rest=STN_rest, noise=noise, activation=clamp )
GPI = Structure( tau=tau, rest=GPI_rest, noise=0.030, activation=clamp )
THL = Structure( tau=tau, rest=THL_rest, noise=noise, activation=clamp )
structures = (CTX, STR, STN, GPI, THL)

CUE = np.zeros(4, dtype=[("mot", float),
                         ("cog", float),
                         ("value" , float),
                         ("reward", float)])
CUE["mot"]    = 0,1,2,3
CUE["cog"]    = 0,1,2,3
CUE["value"]  = 0.5
CUE["reward"] = rewards

def weights(shape):
    N = np.random.normal(0.5, 0.005, shape)
    N = np.minimum(np.maximum(N, 0.0),1.0)
    return (Wmin+(Wmax-Wmin)*N)

W1 = (2*np.eye(4) - np.ones((4,4))).ravel()
W2 = (2*np.eye(16) - np.ones((16,16))).ravel()

connections = {
    "CTX.cog -> STR.cog" : OneToOne( CTX.cog.V, STR.cog.Isyn, weights(4)  ), # plastic (RL)
    "CTX.mot -> STR.mot" : OneToOne( CTX.mot.V, STR.mot.Isyn, weights(4)  ),
    "CTX.ass -> STR.ass" : OneToOne( CTX.ass.V, STR.ass.Isyn, weights(4*4)),
    "CTX.cog -> STR.ass" : CogToAss( CTX.cog.V, STR.ass.Isyn, weights(4)  ),
    "CTX.mot -> STR.ass" : MotToAss( CTX.mot.V, STR.ass.Isyn, weights(4)  ),
    "CTX.cog -> STN.cog" : OneToOne( CTX.cog.V, STN.cog.Isyn, np.ones(4)  ),
    "CTX.mot -> STN.mot" : OneToOne( CTX.mot.V, STN.mot.Isyn, np.ones(4)  ),
    "STR.cog -> GPI.cog" : OneToOne( STR.cog.V, GPI.cog.Isyn, np.ones(4)  ),
    "STR.mot -> GPI.mot" : OneToOne( STR.mot.V, GPI.mot.Isyn, np.ones(4)  ),
    "STR.ass -> GPI.cog" : AssToCog( STR.ass.V, GPI.cog.Isyn, np.ones(4)  ),
    "STR.ass -> GPI.mot" : AssToMot( STR.ass.V, GPI.mot.Isyn, np.ones(4)  ),
    "STN.cog -> GPI.cog" : OneToAll( STN.cog.V, GPI.cog.Isyn, np.ones(4)  ),
    "STN.mot -> GPI.mot" : OneToAll( STN.mot.V, GPI.mot.Isyn, np.ones(4)  ),
    "THL.cog -> CTX.cog" : OneToOne( THL.cog.V, CTX.cog.Isyn, np.ones(4)  ),  # changed
    "THL.mot -> CTX.mot" : OneToOne( THL.mot.V, CTX.mot.Isyn, np.ones(4)  ),  # changed
    "CTX.cog -> THL.cog" : OneToOne( CTX.cog.V, THL.cog.Isyn, np.ones(4)  ),  # changed
    "CTX.mot -> THL.mot" : OneToOne( CTX.mot.V, THL.mot.Isyn, np.ones(4)  ),  # changed
    "CTX.mot -> CTX.mot" : AllToAll( CTX.mot.V, CTX.mot.Isyn, W1,         ),  # new
    "CTX.cog -> CTX.cog" : AllToAll( CTX.cog.V, CTX.cog.Isyn, W1,         ),  # new
    "CTX.ass -> CTX.ass" : AllToAll( CTX.ass.V, CTX.ass.Isyn, W2,         ),  # new
    "CTX.ass -> CTX.cog" : AssToCog( CTX.ass.V, CTX.cog.Isyn, np.ones(4), ), # new (null ?)
    "CTX.ass -> CTX.mot" : AssToMot( CTX.ass.V, CTX.mot.Isyn, np.ones(4), ), # new
    "CTX.cog -> CTX.ass" : CogToAss( CTX.cog.V, CTX.ass.Isyn, np.ones(4)  ), # plastic (Hebbian)
    "CTX.mot -> CTX.ass" : MotToAss( CTX.mot.V, CTX.ass.Isyn, np.ones(4), ), # new (null ?)
    "GPI.cog -> THL.cog" : OneToOne( GPI.cog.V, THL.cog.Isyn, np.ones(4), ), # changed
    "GPI.mot -> THL.mot" : OneToOne( GPI.mot.V, THL.mot.Isyn, np.ones(4), ), # changed
}
for name,gain in gains.items():
    connections[name].gain = gain



# -----------------------------------------------------------------------------
def set_trial(n=2, shuffle=True):
    if shuffle:
        np.random.shuffle(CUE["cog"])
        np.random.shuffle(CUE["mot"])
    CTX.mot.Iext = 0
    CTX.cog.Iext = 0
    CTX.ass.Iext = 0
    for i in range(n):
        c, m = CUE["cog"][i], CUE["mot"][i]
        CTX.mot.Iext[m]     = V_cue + np.random.normal(0,V_cue*noise)
        CTX.cog.Iext[c]     = V_cue + np.random.normal(0,V_cue*noise)
        CTX.ass.Iext[c*4+m] = V_cue + np.random.normal(0,V_cue*noise)


def iterate(dt):
    # Flush connections
    for connection in connections.values():
        connection.flush()

    # Propagate activities
    for connection in connections.values():
        connection.propagate()

    # Compute new activities
    for structure in structures:
        structure.evaluate(dt)


def reset():
    CUE["mot"]    = 0,1,2,3
    CUE["cog"]    = 0,1,2,3
    CUE["value"]  = 0.5
    CUE["reward"] = rewards
    connections["CTX.cog -> STR.cog"].weights = weights(4)
    connections["CTX.cog -> CTX.ass"].weights = np.ones(4)
    reset_activities()

def reset_activities():
    for structure in structures:
        structure.reset()


# def process(n=2, learning=True)
# def learn(reinforcement=True, hebbian=True):

def process(n=2, learning=True):

    # A motor decision has been made
    # The actual cognitive choice may differ from the cognitive choice
    # Only the motor decision can designate the chosen cue
    mot_choice = np.argmax(CTX.mot.V)
    for i in range(n):
        if mot_choice == CUE["mot"][:n][i]:
            cog_choice = CUE["cog"][:n][i]
    choice = cog_choice

    # Compute reward
    reward = np.random.uniform(0,1) < CUE["reward"][choice]

    # Compute prediction error
    error = reward - CUE["value"][choice]

    # Update cues values
    CUE["value"][choice] += error* alpha_CUE

    if learning:

        # Reinforcement learning
        lrate = alpha_LTP if error > 0 else alpha_LTD
        dw = error * lrate * STR.cog.U[choice]
        W = connections["CTX.cog -> STR.cog"].weights
        W[choice] = min(max(W[choice]+dw,Wmin),Wmax)

        # Hebbian learning
        W = connections["CTX.cog -> CTX.ass"].weights
        W += alpha_LTP * np.minimum(CTX.cog.V,10.0)

    return CUE["cog"][:n], choice, reward
