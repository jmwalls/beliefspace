#!/usr/bin/env python
import sys

from beliefspace import *
import matplotlib.pyplot as plt


def scale_quadratic (x,y):
    #return (x + 1.)**2.
    return 1.


if __name__ == '__main__':
    print 'run belief space planning for planar holonomic bot...'

    # initial state
    x0 = np.array ([5., -5.])
    P0 = 3.**2.*np.eye (2)

    # final state
    xf = np.array ([5., 5.])

    # initial control
    nsteps = 25
    #u0 = np.tile ((1./DELTAT)*((xf-x0)/(nsteps-1)).reshape (2,1), (1, nsteps-1))
    u0 = np.zeros ((2,nsteps-1))

    ilqg = Belief_ilqg (x0, P0, u0, xf, scale_quadratic)
    x,P = ilqg.nominal_belief ()
    x_initial = x.copy ()

    # value iteration
    for i in xrange (2):
        ilqg.value_iteration ()
    x,P = ilqg.nominal_belief ()

    print ilqg.ubar


    # plot everything
    fig_xy = plt.figure ()
    ax_xy = fig_xy.add_subplot (111)

    ax_xy.plot (x0[0], x0[1], 'g*', ms=15, mec='g', mew=3)
    ax_xy.plot (xf[0], xf[1], 'r+', ms=15, mew=5)

    ax_xy.plot (x[:,0], x[:,1], 'k.-', lw=2)

    ax_xy.axis ('equal')
    ax_xy.grid ()

    # evaluate measurement model
    axis = ax_xy.axis ()
    X,Y = np.meshgrid (np.linspace (axis[0],axis[1]), 
            np.linspace (axis[2], axis[3]))
    #Z = scale_quadratic (X,Y)
    #ax_xy.contourf (X, Y, Z, cmap=plt.cm.gray_r, 
    #        levels=np.linspace (Z.min (), Z.max (), 25))

    plt.show ()

    sys.exit (0)
