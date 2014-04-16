#!/usr/bin/env python
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
import os.path
import itertools
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import Rectangle
import matplotlib.patheffects as PathEffects

from model import *
from graphics import *
from stimulus import *
from parameters import *
from projections import *

# Target positions (rho,theta)
targets = []
targets_rho   = [2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20] # -> X
targets_theta = [-45, -30, -15, 0, +15, +30, +45]    # -> Y
for i in targets_rho:
    for j in targets_theta:
        targets.append((i,j))


# Decode function
def decode(Z, xmin=+0.0, xmax=+2.0, ymin=-1.0, ymax=+1.0,):
    Y,X = np.mgrid[0:Z.shape[0],0:Z.shape[1]]
    X = xmin + X/float(Z.shape[0]-1)*(xmax-xmin)
    Y = ymin + Y/float(Z.shape[1]-1)*(ymax-ymin)
    Z_sum = Z.sum()
    x = (Z*X).sum() / Z_sum
    y = (Z*Y).sum() / Z_sum
    return x,y


def run(ax, lesion, name, title):

    S = np.zeros((len(targets),4))
    model = Model()
    model.make_lesion(lesion)

    if os.path.exists('lesion-%s.npy' % name):
        S = np.load('lesion-%s.npy' % name)
    else:
        for i,target in enumerate(targets):
            rho,theta = target
            x_,y_ = polar_to_logpolar(rho/90.0, np.pi*theta/180.0)
            x_,y_ = 2*x_, 2*y_-1
            model.reset()
            model.R = stimulus((rho, theta))
            model.run(duration=5*second, dt=5*millisecond, epsilon=0.001)
            x,y = decode(model.SC_V)
            S[i] = x_, y_, x, y
            print 'Target at (%d,%d): %f' % (rho,theta, np.sqrt((x-x_)*(x-x_) + (y-y_)*(y-y_)))
        np.save('lesion-%s.npy' % name, S)

    logpolar_frame(ax)
    for i in range(len(S)):
        ax.plot([S[i,0],S[i,2]],[S[i,1],S[i,3]], color='k', lw=.5)
    ax.scatter(S[:,0], S[:,1], s=35, color='k', marker='o', lw=.5)
    ax.scatter(S[:,2], S[:,3], s=35, color='k', marker='o', facecolors='w', lw=.5)
    ax.text(0,-1, title, ha="left", va="top", fontsize=14)
    if lesion is not None:
        position,size = lesion
        rho,theta = position
        rho,theta = rho/90.0, np.pi*theta/180.0
        x,y = polar_to_logpolar(rho,theta)
        radius = size/90.0
        ax.add_patch(plt.Circle((2*x,2*y-1), radius=radius,  ec='k', fc='w', zorder=+10, alpha=.5))
        ax.add_patch(plt.Circle((2*x,2*y-1), radius=radius,  ec='k', lw=.5, fc='none', zorder=+15))
    #plt.show()


if 0:

    for j in [1, 5, 25, 50, 100, 250, 500]:

        lesion_size =  10
        lesion_rho  =   5
        lesion_theta= -15

        rho = 5
        theta = 0
        model = Model()
        make_lesion(model, [(lesion_rho,lesion_theta), lesion_size])

        dt = 0.01
        model.run(t=j*dt, dt=dt, S = [ [(rho,theta), 1.0, 1.0] ] )

        fig = plt.figure(figsize=(8,6), facecolor='w')
        ax1 = plt.subplot(111, aspect=1)
        logpolar_frame(ax1)
        logpolar_imshow(ax1, model.SCv)

        rho,theta = lesion_rho/90.0, np.pi*lesion_theta/180.0
        x,y = logpolar(rho,theta)
        radius = lesion_size/90.0

        ax1.add_patch(plt.Circle((2*x,2*y-1), radius=radius,  ec='k', fc='w', zorder=+10, alpha=.5))
        ax1.add_patch(plt.Circle((2*x,2*y-1), radius=radius,  ec='k', fc='none', zorder=+15))
        plt.savefig('lesion-after-%03d.pdf' % j)
#        plt.show()


# Error histograms
if 1:
    import matplotlib

    matplotlib.rcParams['xtick.major.width'] = .5
    matplotlib.rcParams['ytick.major.width'] = .5
    matplotlib.rcParams['xtick.direction'] = 'out'
    matplotlib.rcParams['ytick.direction'] = 'out'

    fig = plt.figure(figsize=(20,10))
    fig.patch.set_color('w')
    G = gridspec.GridSpec(2, 3)


    # Intact model
    ax = plt.subplot(G[0, 0], aspect=1)
    run(ax, None, "None", "Intact model")

    # Lesioned model
    ax = plt.subplot(G[1, 0], aspect=1)
    run(ax, [(5,0),12], "(5,0)", "Lesioned model")

    red  = (1,.25,.25)
    blue = (.25,.25,1)

    #  Horizontal errors
    ax = plt.subplot(G[0, 1:])
    ax.set_axisbelow(True)
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    S = np.load('lesion-None.npy').reshape(11,7,4)
    D = np.sqrt( (S[...,2]-S[...,0])**2 +  (S[...,3]-S[...,1])**2)
    X = np.array(targets_rho)
    M = np.mean(D, axis=1) * 100
    E = np.std(D,axis=1) * 100
    ax.bar(X-0.5, M, .5, yerr=E, label='Intact model', color=blue, edgecolor='w', ecolor=blue)

    S = np.load('lesion-(5,0).npy').reshape(11,7,4)
    D = np.sqrt( (S[...,2]-S[...,0])**2 +  (S[...,3]-S[...,1])**2)
    X = np.array(targets_rho)
    M = np.mean(D, axis=1) * 100
    E = np.std(D,axis=1) * 100
    c = (1,.75,.75)
    ax.bar(X, M, .5, yerr=E, label='Intact model', color=red, edgecolor='w', ecolor=red)

    plt.ylim(0,25)
    plt.xlim(0,21)
    plt.xticks([0,10,15,20], [u"0°",u"10°",u"15°",u"20°"])
    plt.yticks([0,5,10,15,20], [u"0%",u"5%",u"10%",u"15%",u"20%"])
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.5',zorder=-10)

    #[t.set_color('0.5') for t in ax.xaxis.get_ticklabels()]
    #[t.set_color('0.5') for t in ax.yaxis.get_ticklabels()]
    [t.set_alpha(0.0) for t in ax.yaxis.get_ticklines()]
    ax.legend(frameon=False, fontsize=12)
    plt.text(0, 21, "Mean encoding error along rosto-caudal axis",
             va='bottom',ha='left',color='k', fontsize=16)

    ax.annotate("Lesion site",
                  xy=(5, 0), xycoords='data',
                  xytext=(5, -2.5), textcoords='data', ha='center',
                  arrowprops=dict(arrowstyle="->", color="k"))


    #  Vertical errors
    ax = plt.subplot(G[1, 1:])
    ax.set_axisbelow(True)
    ax.spines['right'].set_color('none')
    ax.spines['left'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')

    S = np.load('lesion-None.npy').reshape(11,7,4)
    D = np.sqrt( (S[...,2]-S[...,0])**2 +  (S[...,3]-S[...,1])**2)
    X = np.array(targets_theta)
    M = np.mean(D, axis=0) * 100
    E = np.std(D, axis=0) * 100
    ax.bar(X-2.5, M, 2.5, yerr=E, label='Intact model', color=blue, edgecolor='w', ecolor=blue)

    S = np.load('lesion-(5,0).npy').reshape(11,7,4)
    D = np.sqrt( (S[...,2]-S[...,0])**2 +  (S[...,3]-S[...,1])**2)
    X = np.array(targets_theta)
    M = np.mean(D, axis=0) * 100
    E = np.std(D, axis=0) * 100
    ax.bar(X, M, 2.5, yerr=E, label='Intact model', color=red, edgecolor='w', ecolor=red)


    plt.xlim(-50,50)
    plt.xticks([-45,-30,-15,+15,+30,+45],
               [u"-45°",u"-30°",u"-15°", u"15°", u"30°", u"45°"])
    plt.ylim(0,25)
    plt.yticks([0,5,10,15,20], [u"0%",u"5%",u"10%",u"15%",u"20%"])
    ax.grid(which='major', axis='y', linewidth=0.75, linestyle='-', color='0.5',zorder=-10)

    #[t.set_color('0.5') for t in ax.xaxis.get_ticklabels()]
    #[t.set_color('0.5') for t in ax.yaxis.get_ticklabels()]
    [t.set_alpha(0.0) for t in ax.yaxis.get_ticklines()]
    ax.legend(frameon=False, fontsize=12)
    plt.text(-50, 21, "Mean encoding error along vertical axis",
             va='bottom',ha='left',color='k', fontsize=16)
    ax.annotate("Lesion site",
                  xy=(0, 0), xycoords='data',
                  xytext=(0, -2.5), textcoords='data', ha='center',
                  arrowprops=dict(arrowstyle="->", color="k"))

    # # Lesioned model
    # S = np.load('lesion-(5,0).npy').reshape(11,7,4)

    # D = np.sqrt( (S[...,2]-S[...,0])**2 +  (S[...,3]-S[...,1])**2)
    # X = np.array(targets_rho)
    # M = np.mean(D, axis=1) * 100
    # E = np.std(D,axis=1) * 100
    # ax1.bar(X, M, .45, color='r', ecolor='r', yerr=E, label='Lesioned model')


    # X = np.array(targets_theta)
    # M = np.mean(D, axis=0) * 100
    # E = np.std(D, axis=0) * 100
    # ax2.bar(X, M, 4.5,  color='r', ecolor='r', yerr=E, label='Lesioned model')

    # ax1.legend(frameon=False)
    # ax2.legend(frameon=False)

    # ax2.annotate("Lesion site",
    #              xy=(0, 0), xycoords='data',
    #              xytext=(0, -2.5), textcoords='data', ha='center',
    #              arrowprops=dict(arrowstyle="->", color="k"))

    # ax1.set_title("Mean encoding error along rosto-caudal axis")
    # ax1.set_ylabel("Mean encoding error (%)")
    # #ax1.set_xlabel(r"$\rho$ coordinate")

    # ax2.set_title("Mean encoding error along vertical axis")
    # ax2.set_ylabel("Mean encoding error (%)")
    # #ax2.set_xlabel(r"$\phi$ coordinate")

    plt.savefig("fig-lesion.pdf")
    plt.show()
