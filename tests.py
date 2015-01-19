#!/usr/bin/env python
import sys

from beliefspace import *

def func0 (x):
    return np.array ([2.*x[0]**2. + 3.*x[1] + 5.*x[2],
                      x[2]**3.])

def gfunc0 (x):
    return np.array ([[4.*x[0], 3., 5.],
                      [0.,      0., 3.*x[2]**2.]])

def test_jacobian ():
    print 'testing approximate jacobian...',

    x = np.array ([np.sqrt (2)/2, 2., -3.])
    g_approx = approx_jacobian (x, func0, 1e-9, None)
    g_true = gfunc0 (x)

    if np.allclose (g_approx, g_true):
        print 'checks', np.abs (g_approx-g_true).max ()
    else:
        print 'failed'
        print g_approx
        print g_true

def func1 (x):
    return 2.*x[0]**2. + 3.*x[1]**2.*x[2] - x[2]**2.

def gfunc1 (x):
    return np.array ([4.*x[0], 6.*x[1]*x[2], 3.*x[1]**2. - 2.*x[2]])

def hfunc1 (x):
    return np.array ([[4., 0., 0.],
                      [0., 6.*x[2], 6.*x[1]],
                      [0., 6.*x[1], -2.]])

def test_hessian ():
    print 'testing approximate jacobian/hessian...',

    x = np.array ([np.sqrt (2)/2, 2., -3.])
    g_approx, h_approx = approx_jacobian_hessian (x, func1, 1e-6, None)
    g_true, h_true = gfunc1 (x), hfunc1 (x)

    if np.allclose (g_approx, g_true) and np.allclose (h_approx, h_true, atol=1e-1):
        print 'checks', np.abs (h_approx-h_true).max ()
    else:
        print 'failed'
        print g_approx
        print g_true
        print h_approx
        print h_true


def test_beliefs ():
    print 'testing belief packing/unpacking...',

    x = np.array ([np.pi, -np.pi/5])
    P = np.array ([[5., 3.], [3., 3.]])

    bk = belief_pack (x, P)
    xk,Pk = belief_unpack (bk)

    if np.allclose (x, xk) and np.allclose (P, Pk):
        print 'checks'
    else:
        print 'failed'
        print x
        print xk
        print P
        print Pk


if __name__ == '__main__':
    print 'belief space planning unit tests'

    test_jacobian ()
    test_hessian ()
    test_beliefs ()

    sys.exit (0)
