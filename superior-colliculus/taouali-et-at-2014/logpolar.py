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
import os
import numpy as np


def cartesian_to_polar(x, y):
    ''' Cartesian to polar coordinates. '''

    rho = np.sqrt(x**2+y**2)
    theta = np.arctan2(y,x)
    return rho,theta


def polar_to_cartesian(rho, theta):
    ''' Polar to cartesian coordinates. '''

    x = rho*np.cos(theta)
    y = rho*np.sin(theta)
    return x,y


def polar_to_logpolar(rho, theta):
    ''' Polar to logpolar coordinates. '''

    # Shift in the SC mapping function in deg
    A = 3.0
    # Collicular magnification along u axe in mm/rad
    Bx = 1.4
    # Collicular magnification along v axe in mm/rad
    By = 1.8
    xmin, xmax = 0.0, 4.80743279742
    ymin, ymax = -2.76745559565, 2.76745559565
    rho = rho*90.0
    x = Bx*np.log(np.sqrt(rho*rho+2*A*rho*np.cos(theta)+A*A)/A)
    y = By*np.arctan(rho*np.sin(theta)/(rho*np.cos(theta)+A))
    x = (x-xmin)/(xmax-xmin)
    y = (y-ymin)/(ymax-ymin)
    return x, y


def retina_to_colliculus(Rs=(512,256), Cs=(64,64)):
    '''
    Compute the projection indices from retina to colliculus

    Parameters
    ----------

    Rs : (int,int)
        Half-retina shape

    Cs : (int,int)
        Colliculus shape
    '''

    filename = "retina (%d,%d) - colliculus (%d,%d).npy" % (Rs[0],Rs[1],Cs[0],Cs[1])
    if os.path.exists(filename):
        return np.load(filename)

    s = 4
    rho = ((np.logspace(start=0, stop=1, num=s*Rs[1],base=10)-1)/9.)
    theta = np.linspace(start=-np.pi/2,stop=np.pi/2, num=s*Rs[0])

    rho = rho.reshape((s*Rs[1],1))
    rho = np.repeat(rho,s*Rs[0], axis=1)

    theta = theta.reshape((1,s*Rs[0]))
    theta = np.repeat(theta,s*Rs[1], axis=0)

    y,x = polar_to_cartesian(rho,theta)

    xmin,xmax = x.min(), x.max()
    x = (x-xmin)/(xmax-xmin)

    ymin,ymax = y.min(), y.max()
    y = (y-ymin)/(ymax-ymin)

    P = np.zeros((Cs[0],Cs[1],2), dtype=int)
    xi = np.rint(x*(Rs[0]-1)).astype(int)
    yi = np.rint((0.0+1.0*y)*(Rs[1]-1)).astype(int)

    yc,xc = polar_to_logpolar(rho,theta)
    xmin,xmax = xc.min(), xc.max()
    xc = (xc-xmin)/(xmax-xmin)
    ymin,ymax = yc.min(), yc.max()
    yc = (yc-ymin)/(ymax-ymin)
    xc = np.rint(xc*(Cs[0]-1)).astype(int)
    yc = np.rint((.0+yc*1.0)*(Cs[1]-1)).astype(int)

    P[xc,yc,0] = xi
    P[xc,yc,1] = yi
    np.save(filename, P)
    return P


def polar_frame(ax, title=None, legend=False, zoom=False):
    """ Draw a polar frame """

    for rho in [0, 2,5,10,20,40,60,80,90]:
        lw, color, alpha = 1, '0.00', 0.25
        if rho == 90 and not zoom:
            color, lw, alpha = '0.00', 2, 1

        n = 500
        R = np.ones(n)*rho/90.0
        T = np.linspace(-np.pi/2,np.pi/2,n)
        X,Y = polar_to_cartesian(R,T)
        ax.plot(X, Y-1/2, color=color, lw=lw, alpha=alpha)

        if not zoom and rho in [0,10,20,40,80]:
            ax.text(X[-1]*1.0-0.075, Y[-1],u'%d°' % rho, color='k', # size=15,
                    horizontalalignment='center', verticalalignment='center')

    for theta in [-90,-60,-30,0,+30,+60,+90]:
        lw, color, alpha = 1, '0.00', 0.25
        if theta in[-90,+90] and not zoom:
            color, lw, alpha = '0.00', 2, 1
        angle = theta/90.0*np.pi/2

        n = 500
        R = np.linspace(0,1,n)
        T = np.ones(n)*angle
        X,Y = polar_to_cartesian(R,T)
        ax.plot(X, Y, color=color, lw=lw, alpha=alpha)

        if not zoom and theta in [-90,-60,-30,+30,+60,+90]:
            ax.text(X[-1]*1.05, Y[-1]*1.05,u'%d°' % theta, color='k', # size=15,
                    horizontalalignment='left', verticalalignment='center')
    d = 0.01
    ax.set_xlim( 0.0-d, 1.0+d)
    ax.set_ylim(-1.0-d, 1.0+d)
    ax.set_xticks([])
    ax.set_yticks([])

    if legend:
        ax.set_frame_on(True)
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data',-1.2))
        ax.set_xticks([])
        ax.text(0.0,-1.1, "$\longleftarrow$ Foveal",
                  verticalalignment='top', horizontalalignment='left', size=12)
        ax.text(1.0,-1.1, "Peripheral $\longrightarrow$",
                  verticalalignment='top', horizontalalignment='right', size=12)
    else:
        ax.set_frame_on(False)
    if title:
        ax.title(title)


def logpolar_frame(ax, title=None, legend=False):
    """ Draw a log polar frame """

    for rho in [2,5,10,20,40,60,80,90]:
        lw, color, alpha = 1, '0.00', 0.25
        if rho == 90:
            color, lw, alpha = '0.00', 2, 1

        n = 500
        R = np.ones(n)*rho/90.0
        T = np.linspace(-np.pi/2,np.pi/2,n)
        X,Y = polar_to_logpolar(R,T)
        X,Y = X*2, 2*Y-1
        ax.plot(X, Y, color=color, lw=lw, alpha=alpha)
        if rho in [2,5,10,20,40,80]:
            ax.text(X[-1], Y[-1]+0.05, u'%d°' % rho, color='k', # size=15,
                      horizontalalignment='right',  verticalalignment='bottom')

    for theta in [-90,-60,-30, 0, +30,+60,+90]:
        lw, color, alpha = 1, '0.00', 0.25
        if theta in[-90,+90]:
            color, lw, alpha = '0.00', 2, 1
        angle = theta/90.0*np.pi/2

        n = 500
        R = np.linspace(0,1,n)
        T = np.ones(n)*angle
        X,Y = polar_to_logpolar(R,T)
        X,Y = X*2, 2*Y-1
        ax.plot(X,Y, color=color, lw=lw, alpha=alpha)
        ax.text(X[-1]*1.0+.05, Y[-1]*1.0,u'%d°' % theta, color='k', # size=15,
                 horizontalalignment='left', verticalalignment='center')

    d = 0.01
    ax.set_xlim( 0.0-d, 2.0+d)
    ax.set_ylim(-1.0-d, 1.0+d)
    ax.set_xticks([])
    ax.set_yticks([])
    if legend:
        ax.set_frame_on(True)
        ax.spines['left'].set_color('none')
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.spines['bottom'].set_position(('data',-1.2))
        ax.set_xticks([0,2])
        ax.set_xticklabels(['0', '4.8 (mm)'])
        ax.text(0.0,-1.1, "$\longleftarrow$ Rostral",
                  verticalalignment='top', horizontalalignment='left', size=12)
        ax.text(2,-1.1, "Caudal $\longrightarrow$",
                  verticalalignment='top', horizontalalignment='right', size=12)
    else:
        ax.set_frame_on(False)
    if title:
        ax.title(title)


def polar_imshow(axis, Z, *args, **kwargs):
    kwargs['interpolation'] = kwargs.get('interpolation', 'nearest')
    kwargs['cmap'] = kwargs.get('cmap', plt.cm.gray_r)
    kwargs['vmin'] = kwargs.get('vmin', 0)
    kwargs['vmax'] = kwargs.get('vmax', 1)
    kwargs['origin'] = kwargs.get('origin', 'lower')
    axis.imshow(Z, extent=[0,1,-1, 1], *args, **kwargs)



def logpolar_imshow(axis, Z, *args, **kwargs):
    kwargs['interpolation'] = kwargs.get('interpolation', 'nearest')
    kwargs['cmap'] = kwargs.get('cmap', plt.cm.gray_r)
    kwargs['vmin'] = kwargs.get('vmin', 0)
    kwargs['vmax'] = kwargs.get('vmax', 1)
    kwargs['origin'] = kwargs.get('origin', 'lower')
    im = axis.imshow(Z, extent=[0,2,-1, 1], *args, **kwargs)
    # axins = inset_axes(axis, width='25%', height='5%', loc=3)
    # vmin, vmax = Z.min(), Z.max()
    # plt.colorbar(im, cax=axins, orientation='horizontal', ticks=[vmin,vmax], format = '%.2f')
    # axins.xaxis.set_ticks_position('bottom')

# -----------------------------------------------------------------------------
if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon
    from mpl_toolkits.axes_grid1 import ImageGrid
    from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
    from mpl_toolkits.axes_grid1.inset_locator import mark_inset

    P = retina_to_colliculus( (4*1024,4*512), (512,512) )



    # ---------------------------------
    def disc(shape=(1024,1024), center=(512,512), radius = 512):
        ''' Generate a numpy array containing a disc. '''
        def distance(x,y):
            return (x-center[0])**2+(y-center[1])**2
        D = np.fromfunction(distance,shape)
        return np.where(D<radius*radius,1.0,0.0)
    # Checkerboard pattern for retina
    grid = 2*32
    even = grid / 2 * [0, 1]
    odd = grid / 2 * [1, 0]
    R = np.row_stack(grid / 2 * (even, odd))
    R = R.repeat(grid, axis=0).repeat(grid, axis=1)
    # Mask with a disc
    R = R * disc((4*1024,4*1024), (4*512,4*512), 4*512)
    # Take half-retina
    R = R[:,4*512:]
    # Project to colliculus
    SC = R[P[...,0], P[...,1]]


    fig = plt.figure(figsize=(10,8), facecolor='w')
    ax1, ax2 = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.5)
    polar_frame(ax1, legend=True)
    polar_imshow(ax1, R, vmin=0, vmax=5)
    logpolar_frame(ax2, legend=True)
    logpolar_imshow(ax2, SC, vmin=0, vmax=5)
    plt.savefig("retina-colliculus-checkboard.pdf")
    plt.show()


    # ---------------------------------
    fig = plt.figure(figsize=(10,8), facecolor='w')
    ax1, ax2 = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.5)
    radius = 0.01

    polar_frame(ax1, legend=True)
    def draw(X,Y):
        poly = Polygon(zip(X,Y), facecolor='b', edgecolor='none', alpha=0.1)
        ax1.add_patch(poly)
        ax1.plot(X,Y, c='k', lw=.5, alpha=.5)

    rho = 30/90.0
    theta  = 0
    for radius in np.linspace(.005,.3,10):
        x0,y0 = polar_to_cartesian(rho,theta)
        T = np.linspace(0,2*np.pi,1000)
        X = x0 + radius*np.cos(T)
        Y = y0 + radius*np.sin(T)
        draw(X,Y)

    logpolar_frame(ax2, legend=True)
    def draw(X,Y):
        R,T = cartesian_to_polar(X,Y)
        X,Y = polar_to_logpolar(R,T)
        X = 2*X
        Y = 2*Y-1
        poly = Polygon(zip(X,Y), facecolor='r', edgecolor='none', alpha=0.1)
        ax2.add_patch(poly)
        ax2.plot(X,Y, c='k', lw=.5, alpha=.5)

    rho = 30/90.0
    theta  = 0
    for radius in np.linspace(.005,.3,10):
        x0,y0 = polar_to_cartesian(rho,theta)
        T = np.linspace(0,2*np.pi,1000)
        X = x0 + radius*np.cos(T)
        Y = y0 + radius*np.sin(T)
        draw(X,Y)

    plt.savefig("retina-colliculus-circles.pdf")
    plt.show()


    # ---------------------------------
    fig = plt.figure(figsize=(10,8), facecolor='w')
    ax1, ax2 = ImageGrid(fig, 111, nrows_ncols=(1,2), axes_pad=0.5)


    def stimulus(shape, position, size):
        x,y = polar_to_cartesian(position[0]/90.0, np.pi*position[1]/180.0)
        Y,X = np.mgrid[0:shape[0],0:shape[1]]
        X = X/float(shape[1])
        Y = 2*Y/float(shape[0])-1
        r = (X-x)**2+(Y-y)**2
        return np.exp(-0.5*r/(size/90.0))


    polar_frame(ax1, legend=True)

    zax = zoomed_inset_axes(ax1, 6, loc=1)
    polar_frame(zax, zoom=True)
    zax.set_xlim(0.0, 0.1)
    zax.set_xticks([])
    zax.set_ylim(-.05, .05)
    zax.set_yticks([])
    zax.set_frame_on(True)
    mark_inset(ax1, zax, loc1=2, loc2=4, fc="none", ec="0.5")


    R  = stimulus((4*1024,4*512), (1,0), 0.005)
    R += stimulus((4*1024,4*512), (5,0), 0.005)
    polar_imshow(ax1, R)
    polar_imshow(zax, R)

    logpolar_frame(ax2, legend=True)
    SC = R[P[...,0], P[...,1]]
    logpolar_imshow(ax2, SC)

    plt.savefig("retina-colliculus-stimulus.pdf")
    plt.show()
