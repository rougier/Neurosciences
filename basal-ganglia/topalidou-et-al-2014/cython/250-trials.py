# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2014, Nicolas P. Rougier
# Distributed under the (new) BSD License.
#
# Contributors: Nicolas P. Rougier (Nicolas.Rougier@inria.fr)
# -----------------------------------------------------------------------------
# References:
#
# * Interaction between cognitive and motor cortico-basal ganglia loops during
#   decision making: a computational study. M. Guthrie, A. Leblois, A. Garenne,
#   and T. Boraud. Journal of Neurophysiology, 109:3025–3040, 2013.
# -----------------------------------------------------------------------------
import numpy as np
from model import *
from display import *


def debug(time, cues, choice, reward):
    n = len(cues)
    cues = np.sort(cues)

    R.append(reward)
    if choice == cues[0]:
        P.append(1)
    else:
        P.append(0)

    print "Choice:         ",
    for i in range(n):
        if choice == cues[i]:
            print "[%d]" % cues[i],
        else:
            print "%d" % cues[i],
        if i < (n-1):
            print "/",
    if choice == cues[0]:
        print " (good)"
    else:
        print " (bad)"

    print "Reward (%3d%%) :   %d" % (int(100*CUE["reward"][choice]),reward)
    print "Mean performance: %.3f" % np.array(P).mean()
    print "Mean reward:      %.3f" % np.array(R).mean()
    print "Response time:    %d ms" % (time)
    print "CTX.cog->CTX.ass:", connections["CTX.cog -> CTX.ass"].weights
    print


n_experiments = 250
n_trials      = 120
P = np.zeros((n_experiments,n_trials))
filename = "%d-experiments-%d-trials-performances.npy" % (n_experiments, n_trials)

# Put 1 if you want to run a new set of experiments
if 0:
    for k in range(n_experiments):
        reset()
        p = []
        for j in range(n_trials):
            reset_activities()
            # Settling phase (500ms)
            for i in xrange(0,500):
                iterate(dt)
            # Trial setup
            set_trial(n=2)
            # Learning phase (2500ms)
            for i in xrange(500,3000):
                iterate(dt)
                # Test if a decision has been made
                if CTX.mot.delta > decision_threshold:
                    cues, choice, reward = process(n=2, learning=True)
                    cues = np.sort(cues)
                    if choice == cues[0]:
                        p.append(1)
                    else:
                        p.append(0)
                    P[k,j] = np.mean(p)
                    break
        print "Experiment %d: %g" % (k,np.mean(p))
    np.save(filename, P)

P = np.load(filename)


from matplotlib import rcParams
rcParams['xtick.direction'] = 'out'
rcParams['ytick.direction'] = 'out'

plt.figure(figsize=(12,6), dpi=72, facecolor="white")
axes = plt.subplot(111)
axes.spines['right'].set_color('none')
axes.spines['top'].set_color('none')
axes.xaxis.set_ticks_position('bottom')
axes.spines['bottom'].set_position(('data',0))
axes.yaxis.set_ticks_position('left')

p = np.round(np.mean(P,axis=0)[-1],2)

axes.axhline(p, color = '.25', linewidth=.5, linestyle="-")
axes.axhline(.5, color = '.5', linewidth=1, linestyle="--")
axes.text(120,.51,"Chance level", ha="right", va="bottom", color='.5', fontsize=16)
axes.plot(1+np.arange(n_trials), np.mean(P,axis=0), lw=2, c='k')

yticks = np.sort(np.append(np.linspace(0,1,6,endpoint=True), p))
plt.yticks(yticks)

plt.xlabel("Trial number")
plt.xlim(0,120)
plt.ylabel("Proportion of optimum trials")
plt.ylim(0,1)

plt.show()
