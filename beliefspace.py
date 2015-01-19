"""
belief space iLQG

Gaussian beliefs are represented by their mean vector and the cholesky factor
of the covariance matrix
"""
import numpy as np
from scipy.linalg import sqrtm

NSTATE = 2
NBEL = 5 # 2 state dimension + 3 covariance
NCONT = 2

DELTAT = 0.5


class Multivariate_gaussian (object):
    """
    multivariate Gaussian class used to produce samples from a multivariate
    Gaussian distribution.
    """
    def __init__ (self, mu, Sigma):
        self.mu = mu.copy ()
        self.Sigma = Sigma.copy ()

        self._dim = len (self.mu)
        self._sqrtm = np.linalg.cholesky (Sigma)

    def sample (self):
        s = np.random.randn (self._dim)
        return self.mu + self._sqrtm.dot (s)


def approx_jacobian (x, func, epsilon, f0, *args):
    """
    approximate and Jacobian matrix of a vector-valued function by forward
    differences.

    Parameters
    -----------
    x : array_like, vector at which to evaluate Jacobian
    func : callable f (x,*args), vector-valued function
    epsilon : float, perturbation for difference
    f0 : float or None, function evaluated at x, else None
    args : sequence, additional arguments passed to func.
    
    Returns
    --------
    An array of dimensions ``(lenf, lenx)`` where ``lenf`` is the length of
    the outputs of `func`, and ``lenx`` is the number of elements in `x`.

    Notes
    ------
    modified from scipy.optimize.slsqp's approx_jacobian
    """
    x0 = np.asfarray (x)
    if f0 is None:
        f0 = np.atleast_1d (func (*((x0,)+args) ))

    n,m = len (x0), len (f0)
    jac = np.zeros ((n, m))
    dx = np.zeros (n)
    
    for i in range (n):
        dx[i] = epsilon
        jac[i] = (func (*((x0+dx,)+args)) - f0)/epsilon
        dx[i] = 0.0
    
    return jac.transpose ()


def approx_jacobian_hessian (x, func, epsilon, f0, *args):
    """
    approximate Jacobian and Hessian of scalar-valued function by central
    differences.

    Parameters
    -----------
    x : array_like, vector at which to evaluate derivatives
    func : callable f (x, *args), scalar-valued function
    epsilon : float, perturbation for difference
    f0 : float or None, function evaluated at x, else None
    args : sequence, additional arguments passed to func

    Returns
    --------
    jac : array of dimensions ``(lenx,)'' where ``lenx'' is the number of
        elements in `x`
    hes : array of dimensions ``(lenx, lenx)''
    """
    x0 = np.asfarray (x)
    if f0 is None:
        f0 = func (*((x0,)+args))
    epsilon2 = epsilon**2.

    n = len (x0)
    jac, hess = np.zeros (n), np.zeros ((n, n))
    dx = np.zeros (n)

    for i in range (n):
        dx[i] = epsilon
        fp, fm = func (*((x0+dx,)+args)), func (*((x0-dx,)+args))
        jac[i] = (fp - fm)/(2.*epsilon)
        hess[i,i] = (fp + fm - 2.*f0)/epsilon2
        dx[i] = 0.0

        for j in range (0,i):
            dx[i], dx[j] = epsilon, epsilon
            fpp = func (*((x0+dx,)+args))
            fmm = func (*((x0-dx,)+args))

            dx[i], dx[j] = epsilon, -epsilon
            fpm = func (*((x0+dx,)+args))
            fmp = func (*((x0-dx,)+args))

            hess[i,j] = 0.25*((fpp + fmm - fmp - fpm)/epsilon2)
            hess[j,i] = hess[i,j]
            dx[i], dx[j] = 0.0, 0.0

    return jac, hess


def belief_pack (xk, Pk):
    """
    convenience function to pack Gaussian mean/cov into vectorized belief
    """
    Lk = sqrtm (Pk)
    return np.array ([xk[0], xk[1], Lk[0,0], Lk[1,0], Lk[1,1]])


def belief_unpack (bk):
    """
    convenience function to unpack Gaussian mean/cov from vectorized belief
    """
    xk = bk[:2]
    Lk = np.array ([[bk[2], bk[3]], 
                    [bk[3], bk[4]]])
    return xk, Lk.dot (Lk)


class Belief_ilqg (object):
    """
    planning instance

    Parameters
    -----------
    nsteps : number of steps
    xf : goal state

    ubar : current nominal control trajectory
    L : current feedback control policy
    l : current feedforward control policy

    bbar : current nominal belief trajectory

    Rc : control stage cost matrix
    Qc : state stage cost matrix
    Ql : state terminal cost matrix
    """
    def __init__ (self, x0, P0, u0, xf, env):
        """
        Parameters
        -----------
        x0/P0 : initial belief
        u0 : initial nominal control
        xf : goal state
        env : function mapping x,y to measurement noise
        """
        self.nsteps = u0.shape[1]+1
        self.xf = xf.copy ()

        self.noise_environment = env

        # cost weight matrices
        self.Rc = np.eye (NCONT)
        self.Qc = np.eye (NSTATE)
        self.Ql = 10.*self.nsteps*self.Qc

        # compute initial nominal trajectory
        self.ubar = u0.copy ()
        self.L = [np.zeros ((NCONT,NBEL))]*(self.nsteps-1)
        self.l = [np.zeros (NCONT)]*(self.nsteps-1)

        self.b0 = belief_pack (x0, P0)
        self.bbar = np.zeros ((NBEL,self.nsteps))
        self.bbar, ubar = self._compute_nominal (self.L, self.l)

        self.cost = np.inf

    def nominal_belief (self):
        x,P = [], []
        for k in xrange (self.nsteps):
            xk,Pk = belief_unpack (self.bbar[:,k])
            x.append (xk)
            P.append (Pk)
        return np.asarray (x), np.asarray (P)

    def noise_control (self, uk):
        return np.array ([[uk[0]**2., 0.],
                          [0., uk[1]**2.]])

    def _cost_stage (self, bk, uk):
        c_control = uk.T.dot (self.Rc).dot (uk)

        xk,Pk = belief_unpack (bk)
        c_belief = np.trace (Pk.dot (self.Qc))

        c_collision = 0.
        return c_control + c_belief + c_collision

    def _cost_stage_vector (self, vk):
        return self._cost_stage (vk[:NBEL], vk[NBEL:])

    def _cost_terminal (self, bk):
        xk,Pk = belief_unpack (bk)
        c_state = xk.T.dot (self.Ql).dot (xk)
        c_belief = np.trace (Pk.dot (self.Ql))
        return c_state + c_belief

    def _dynamics_g (self, bk, uk):
        """
        compute belief update for 2d holonomic car

        Parameters
        -----------
        bk : belief
        uk : control

        Returns
        --------
        bk+1 : g (bk, uk)
        """
        xk,Pk = belief_unpack (bk)

        xkn = xk + DELTAT*uk

        Mk = self.noise_control (uk)
        Nk = self.noise_environment (xk[0], xk[1])*np.eye (2)
        Gk = Pk + Mk.dot (Mk.T)
        Kk = Gk.dot (np.linalg.inv (Gk + Nk.dot (Nk.T)))
        Pkn = Gk - Kk.dot (Gk)

        return belief_pack (xkn, Pkn)

    def _dynamics_vector_g (self, vk):
        return self._dynamics_g (vk[:NBEL], vk[NBEL:])

    def _dynamics_W (self, bk, uk):
        """
        compute belief update covariance for 2d holonomic car

        Parameters
        -----------
        bk : belief
        uk : control

        Returns
        --------
        Wk : covariance of belief
        """
        xk,Pk = belief_unpack (bk)

        Mk,Nk = np.eye (2), np.eye (2) 
        Gk = Pk + Mk.dot (Mk.T)
        Kk = Gk.dot (np.linalg.inv (Gk + Nk.dot (Nk.T)))

        return sqrtm (Kk.dot (Gt))

    def _dynamics_vector_W (self, vk):
        return self._dynamics_W (vk[:NBEL], vk[NBEL:])

    def _compute_nominal (self, L, l, eps=1.):
        """
        compute nominal belief/control trajectory given control policy defined
        by feedback matrix L and feedforward term l

        Parameters
        -----------
        L : nsteps-1 length list of NBELxNBEL array_like
        l : nsteps-1 length list of NBEL array_like

        Returns
        --------
        bbar : update nominal belief trajectory
        ubar : updated nominal control trajectory
        """
        bbar = np.zeros (self.bbar.shape)
        ubar = np.zeros (self.ubar.shape)
        bbar[:,0] = self.b0
        for k in xrange (0, self.nsteps-1):
            ubar[:,k] = L[k].dot (bbar[:,k] - self.bbar[:,k]) + eps*l[k] + self.ubar[:,k]
            bbar[:,k+1] = self._dynamics_g (bbar[:,k], ubar[:,k])
        return bbar, ubar

    def _value_iteration_step (self, k, Sk, sk, ssk):
        """
        perform single value iteration step
        """
        bk, uk = self.bbar[:,k], self.ubar[:,k]

        # 1. linearize dynamics about nominal
        vbar = np.hstack ((bk, uk))
        F = approx_jacobian (vbar, self._dynamics_vector_g, 1e-6, None)
        Fk = F[:,:NBEL]
        Gk = F[:,NBEL:]

        Fki = None
        Gki = None
        ebarki = None

        # 2. quadratice stage costs about nominal
        cbark = self._cost_stage (bk, uk)
        qrk, QRk = approx_jacobian_hessian (vbar,
                self._cost_stage_vector, 1e-6, cbark)
        Qk = QRk[:NBEL,:NBEL]
        Rk = QRk[NBEL:,NBEL:]
        Pk = QRk[:NBEL,NBEL:]

        qk = qrk[:NBEL]
        rk = qrk[NBEL:]

        # 3. compute control policy and value function
        Ck = Qk + Fk.T.dot (Sk).dot (Fk) + Fki.T.dot (Sk).dot (Fki)
        Dk = Rk + Gk.T.dot (Sk).dot (Gk) + Gki.T.dot (Sk).dot (Gki)
        Ek = Pk + Fk.T.dot (Sk).dot (Gk) + Fki.T.dot (Sk).dot (Gki)

        ck = qk + Fk.T.dot (sk) + Fki.T.dot (Sk).dot (ebarki)
        dk = rk + Gk.T.dot (sk) + Gki.T.dot (Sk).dot (ebarki)
        ek = cbark + ssk + ebarki.T.dot (Sk).dot (ebarki)

        Dinv = np.linalg.inv (Dk)
        Lk = -Dinv.dot (Ek.T)
        lk = -Dinv.dot (dk)

        Sk = Ck - Ek.dot (Dinv).dot (Ek.T)
        sk = cbark - Ek.dot (Dinv).dot (dk)
        ssk = ek - (1./2)*dk.T.dot (Dinv).dot (dk)

        return Lk, lk, Sk, sk, ssk

    def value_iteration (self):
        """
        refine current nominal path and control
        """
        # quadratize final state cost
        ssk = self._cost_terminal (self.bbar[:,-1])
        Sk, sk = approx_jacobian_hessian (self.bbar[:,-1],
                self._cost_terminal, 1e-6, ssk)

        # backward recursion starting with step N-1 (N-2 for 0 indexed)
        self.L
        for k in xrange (self.nsteps-2, -1, -1):
            Lk, lk, Sk, sk, ssk = self._value_iteration_step (k, Sk, sk, ssk)
            L.append (Lk)
            l.append (lk)

        # update nominal trajectory
        # backtracking linesearch
        # u = L (b - bbar) + eps l + ubar
        bbar, ubar = self._compute_nominal (L, l, eps=1.)

        self.bbar = bbar
        self.ubar = ubar
        self.L = L
        self.l = l

        return None

