# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
import random
import numpy as np
from model import *
from display import *

# C1: Familiar cues, GPI on

reset()
CUE["reward"] = 0.75, 0.25, 0.75, 0.25

n = 100

C1 = np.zeros((n,120), dtype=[("P", float), ("RT", float)])

for k in range(n):

    trained = False
    while not trained:
        reset()
        CUE["reward"] = 0.75, 0.25, 0.75, 0.25
        connections["GPI.cog -> THL.cog"].active = True
        connections["GPI.mot -> THL.mot"].active = True
        P = []
        for j in range(25):
            reset_activities()
            for i in xrange(0,500):
                iterate(dt)
            if random.uniform(0,1) < 0.5: CUE["cog"] = 0,1,2,3
            else:                         CUE["cog"] = 1,0,2,3
            set_trial(n=2, mot_shuffle=True, cog_shuffle=False)
            for i in xrange(500,3000):
                iterate(dt)
                if CTX.mot.delta > decision_threshold:
                    cues, choice, reward = process(n=2, learning=(True,True))
                    if choice == 0:
                        P.append(1)
                    else:
                        P.append(0)
                    break
        if np.mean(P) > .75:
            trained = True

    connections["GPI.cog -> THL.cog"].active = True
    connections["GPI.mot -> THL.mot"].active = True

    for j in range(120):
        reset_activities()
        for i in xrange(0,500):
            iterate(dt)
        # CUE["cog"] = 0,1,2,3
        if random.uniform(0,1) < 0.5:  CUE["cog"] = 0,1,2,3
        else:                          CUE["cog"] = 1,0,2,3
        set_trial(n=2, mot_shuffle=True, cog_shuffle=False)
        for i in xrange(500,3000):
            iterate(dt)
            if CTX.mot.delta > decision_threshold:
                cues, choice, reward = process(n=2, learning=True)
                cues = np.sort(cues)
                if choice == 0:
                    C1["P"][k,j] = 1
                C1["RT"][k,j] = i-500
                break
    print "Experiment %d: %g" % (k, np.mean(C1["P"][k]))

np.save("Piron-C1.npy",C1)
