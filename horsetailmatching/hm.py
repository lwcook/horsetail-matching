import numpy as np
import copy
import pdb
import random as rdm
import time
import scipy.special as scp
import scipy.stats as scs
import matplotlib.pyplot as plt
from scipy import optimize as scipyopt

import utilities as utils


class HorsetailMatching():
    '''Class for using horsetail matching within an optimization. The main
    functionality is to create functions that return the horsetail matching
    metric (and optionally its gradient) that can be used with external
    optimizers.

    :param function fqoi: function that returns the quantity of interest, it
        must take two ordered arguments - the value of the design variable
        vector and the value of the uncertainty vector.
    :param list uncertain_parameters_params: list of Parameter objects that
        describe the uncertain inputs for the problem.
    :param function ftarg: function that returns the value of the target
        inverse CDF given a value in [0,1]. [default t(h) = 0]
    :param function ftarg_u: under mixed uncertainties, function that returns
        the value of the target for the upper bound horsetail curve given a
        value in [0,1]. [default t_u(h) = 0]
    :param function ftarg_l: under mixed uncertainties, function that returns
        the value of the target for the lower bound horsetail curve given a
        value in [0,1]. [default t_u(h) = 0]
    :param int n_samples_prob: number of samples to take from the
        probabilsitic uncertainties.
    :param int n_samples_int: number of samples to take from the
        interval uncertainties.
    '''

    def __init__(self, fqoi, uncertain_parameters,
            ftarg=None, ftarg_u=None, ftarg_l=None,
            n_samples_prob=100, n_samples_int=10,
            n_integration_points=100,
            method='empirical',
            q_low=0, q_high=10, bw=None, alpha=100):

        self.fqoi = fqoi
        self.u_params = utils.makeIter(uncertain_parameters)

        # Internally split uncertainties into interval and prob
        # Uncertainties are stored as a tuple of index and param
        self.u_int, self.u_prob = [], []
        for ii, u in enumerate(self.u_params):
            if u.distribution == 'interval':
                self.u_int.append((ii, u))
            else:
                self.u_prob.append((ii, u))

        # Target function can be provided as a single CDF for probabilistic
        # uncertainties, or as upper and lower targets for mixed uncertainties
        if ftarg is None:
            self.ftarg = lambda h: 0.

        if ftarg_u is None and ftarg_l is None:
            self.ftarg_u = self.ftarg
            self.ftarg_l = self.ftarg

        self.method = method
        self.M_prob = n_samples_prob
        self.M_int = n_samples_int

        self.N_quad = n_integration_points
        self.q_low = q_low
        self.q_high = q_high
        self.bw = bw
        self.alpha = alpha

        self.bSAA = True
        self.stored_u_samples = None

        self._checkAttributes()

    def evalMetric(self, x, method=None):
        '''Evaluates the horsetail matching metric at given values of the
        design variables.

        :param iterable x: values of the design variables
        :param str method: method to use to evaluate the metric
        '''
        if method is None:
            method = self.method

        # Make sure everything is in order
        self._checkAttributes()

        # Array of shape (M_int, M_prob, N_uncertainties)
        u_samples = self._getParameterSamples()

        # Array of shape (M_int, M_prob)
        q_samples = np.zeros(u_samples.shape[0:2])
        for ii in np.arange(q_samples.shape[0]):
            for jj in np.arange(q_samples.shape[1]):
                q_samples[ii, jj] = self.fqoi(x, u_samples[ii, jj])

        if self.bw is None:
            self.bw = (4/(3.*q_samples.shape[1]))**(1/5.)*np.std(q_samples[0,:])

        if method.lower() == 'empirical':
            return self._evalMetricEmpirical(q_samples)
        elif method.lower() == 'kernel':
            return self._evalMetricKernel(q_samples)
        elif method.lower() == 'gradient':
            return self._evalGradientKernel(q_samples)
        else:
            raise ValueError('Unsupported metric evalation method')

    def plotHorsetail(self, *plotargs, **plotkwargs):
        ql, qu, hl, hu = self._ql, self._qu, self._hl, self._hu
        qh, hh = self._qh, self._hh

        ql, hl = self._makePlotArrays(ql, hl)
        qu, hu = self._makePlotArrays(qu, hu)

        for qi, hi in zip(qh, hh):
            plt.plot(qi, hi, c='grey', alpha=0.5, lw=0.5)
        plt.plot(ql, hl, *plotargs, **plotkwargs)
        plt.plot([self.ftarg_l(hi) for hi in hl], hl, 'b:')
        plt.plot(qu, hu, *plotargs, **plotkwargs)
        plt.plot([self.ftarg_u(hi) for hi in hu], hu, 'r:')
        plt.ylim([0,1])

##############################################################################
##  PRIVATE METHODS  ##
##############################################################################

    def _evalMetricEmpirical(self, q_samples):

        h_horsetail = np.zeros(q_samples.shape)
        q_horsetail = np.zeros(q_samples.shape)
        for ii in np.arange(q_samples.shape[0]):
            q_horsetail[ii, :], h_horsetail[ii, :] = \
                    _getECDFfromSamples(q_samples[ii, :])

        if q_samples.shape[0] > 1:
            q_u = q_horsetail.min(axis=0)
            q_l = q_horsetail.max(axis=0)
        else:
            q_u, q_l = q_horsetail[0], q_horsetail[0]
        h_u, h_l = h_horsetail[0], h_horsetail[0]  # h is same for all ECDFs

        D_u, D_l = 0., 0.
        for (qui, hui), (qli, hli) in zip(zip(q_u, h_u), zip(q_l, h_l)):
            D_u += (1./self.M_prob)*(qui - self.ftarg_u(hui))**2
            D_l += (1./self.M_prob)*(qli - self.ftarg_l(hli))**2

        dhat = np.sqrt(D_u + D_l)
        self._ql, self._qu, self._hl, self._hu = q_l, q_u, h_l, h_u
        self._qh, self._hh = q_horsetail, h_horsetail
        return dhat

    def _evalMetricKernel(self, q_samples):

        h_horsetail = np.zeros([q_samples.shape[0], self.N_quad])
        q_horsetail = np.zeros([q_samples.shape[0], self.N_quad])
        qis = np.linspace(self.q_low, self.q_high, self.N_quad)

        for ii in np.arange(q_samples.shape[0]):
            qjs = q_samples[ii, :]
            rmat = qis.reshape([self.N_quad, 1])-qjs.reshape([1, self.M_prob])
            Kcdf = _kernel(rmat, self.M_prob, bw=self.bw, bGrad=False)

            h_horsetail[ii, :] = Kcdf.dot(np.ones([self.M_prob, 1])).flatten()
            q_horsetail[ii, :] = qis

        h_u = np.array([_extalg(h_horsetail[:, iq], self.alpha) for iq in
            np.arange(self.N_quad)])
        h_l = np.array([_extalg(h_horsetail[:, iq], -self.alpha) for iq in
            np.arange(self.N_quad)])

        t_u = np.array([self.ftarg_u(hi) for hi in h_u])
        t_l = np.array([self.ftarg_l(hi) for hi in h_l])

        D_u = _matrix_integration(qis, h_u, t_u)
        D_l = _matrix_integration(qis, h_l, t_l)
        dhat = float(np.sqrt(D_u + D_l))

        self._ql, self._qu, self._hl, self._hu = qis, qis, h_l, h_u
        self._qh, self._hh = q_horsetail, h_horsetail
        return dhat

    def _evalGradientKernel(self, x):
        pass

    def _getParameterSamples(self):
        '''Returns a 3D array of size (num_interval_samples, num_prob_samples,
        num_uncertainties). If there are no interval or probabilistic
        uncertainties, then the corresponding dimensions are 1'''

        if self.bSAA and self.stored_u_samples is not None:
            return self.stored_u_samples
        else:
            N_u = len(self.u_int) + len(self.u_prob)

            u_samples = np.zeros(self._u_sample_dim)

            # Sample over interval uncertainties, and then at each sampled
            # value sample over the probabilistic uncertainties
            for ii in np.arange(u_samples.shape[0]):
                u_i = _getSample(self.u_int, N_u)

                u_sub = np.zeros([u_samples.shape[1], N_u])
                for jj in np.arange(u_samples.shape[1]):
                    u_sub[jj,:] = u_i + _getSample(self.u_prob, N_u)

                u_samples[ii,:,:] = u_sub

            self.stored_u_samples = u_samples
            return u_samples


    def _checkAttributes(self):

        if len(self.u_int) == 0 and len(self.u_prob) == 0:
            raise ValueError('No uncertain parameters provided')

        N_u = len(self.u_int) + len(self.u_prob)

        # Mixed uncertainties
        if len(self.u_int) > 0 and len(self.u_prob) > 0:
            self._u_sample_dim = (self.M_int, self.M_prob, N_u)

        # Probabilistic uncertainties
        elif len(self.u_int) == 0:
            self.M_int = 1
            self._u_sample_dim = (1, self.M_prob, N_u)

        # Interval Uncertainties
        elif len(self.u_prob) == 0:
            self.M_prob = 1
            self._u_sample_dim = (self.M_int, 1, N_u)
            self.bw = 1e-3

    def _makePlotArrays(self, q, h):
        q = np.insert(q, 0, q[0])
        h = np.insert(h, 0, 0)
        q = np.insert(q, 0, self.q_low)
        h = np.insert(h, 0, 0)
        q = np.append(q, q[-1])
        h = np.append(h, 1)
        q = np.append(q, self.q_high)
        h = np.append(h, 1)
        return q, h


##############################################################################
##  PRIVATE FUNCTIONS
##############################################################################

def _getSample(u_params, N_u):
    '''Function that samples only the uncertainties specified in u_params,
    and returns a vectors with the indices given by u_params filled with
    sampled values of the uncertainties'''
    vu = np.zeros(N_u)
    if len(u_params) > 0: # Return zeros if no parameters given
        for (i, u) in u_params:
            vu[i] = (u.getSample())
    return vu

def _getECDFfromSamples(q_samples):
    vq = np.sort(q_samples)
    M = len(q_samples)
    vh = [(1./M)*(0.5 + j) for j in range(M)]
    return vq, vh

def _kernel(points, M=None, bw=None, ktype='gauss', bGrad=False):

    if M is None:
        M = np.array(points).size
    if bw is None:
        bw = (4./(3.*M))**(1./5.)*np.std(points)

    # NB make evaluations matrix compatible
    if ktype == 'gauss' or ktype == 'gaussian':
        KernelMat = (1./M)*scp.ndtr(points/bw)
    elif ktype == 'gemp':
        bwemp = bw/100.
        KernelMat = (1./M)*scp.ndtr(points/bwemp)
    elif ktype == 'step' or ktype == 'empirical':
        KernelMat = (1./M)*step(points)
    elif ktype == 'uniform' or ktype == 'uni':
        KernelMat = (1./M)*ramp(points, width=bw*np.sqrt(12))
    elif ktype == 'triangle' or ktype == 'tri':
        KernelMat = (1./M)*trint(points, width=bw*2.*np.sqrt(6))

    if bGrad:
        if ktype == 'gauss' or ktype == 'gaussian':
            const_term = 1.0/(M * np.sqrt(2*np.pi*bw**2))
            KernelGradMat = const_term * np.exp(-(1./2.) * (points/bw)**2)
        elif ktype == 'gemp':
            const = 1.0/(M * np.sqrt(2*np.pi*bwemp**2))
            KernelGradMat = const * np.exp(-(1./2.) * (points/bwemp)**2)
        elif ktype == 'uniform' or ktype == 'uni':
            width = bw*np.sqrt(12)
            const = (1./M)*(1./width)
            KernelGradMat = const*(step(points+width/2) -
                                   step(points-width/2))
        elif ktype == 'triangle' or ktype == 'tri':
            width = bw*2.*np.sqrt(6)
            const = (1./M)*(2./width)
            KernelGradMat = const*(ramp(points+width/4, width/2) -
                                   ramp(points-width/4, width/2))
        else:
            KernelGradMat = 0*points
            print('Warning: kernel type gradient not supported')

        return KernelMat, KernelGradMat
    else:
        return KernelMat

def _extalg(xarr, alpha=100):
    '''Given an array xarr of values, smoothly return the max/min'''
    return sum(xarr * np.exp(alpha*xarr))/sum(np.exp(alpha*xarr))

def _extgrad(xarr, alpha=100):
    '''Given an array xarr of values, return the gradient of the smooth min/max
    swith respect to each entry in the array'''
    term1 = np.exp(alpha*xarr)/sum(np.exp(alpha*xarr))
    term2 = 1 + alpha*(xarr - extalg(xarr, alpha))

    return term1*term2

def _matrix_integration(q, h, t):
    ''' Returns the dp metric for a single horsetail
    curve at a given value of the epistemic uncertainties'''

    N = len(q)

    # correction if CDF has gone out of trapezium range
    if h[-1] < 0.9: h[-1] = 1.0

    W = np.zeros([N, N])
    for i in range(N):
        W[i, i] = 0.5*(h[min(i+1, N-1)] - h[max(i-1, 0)])

    dp = (q - t).T.dot(W).dot(q - t)

    return dp

def _matrix_gradient(q, h, t, h_dx):
    ''' Returns the gradient with respect to a single variable'''

    N = len(q)
    W = np.zeros([N, N])
    Wprime = np.zeros([N, N])
    for i in range(N):
        W[i, i] = 0.5*(h[min(i+1, N-1)] - h[max(i-1, 0)])
        Wprime[i, i] = \
            0.5*(h_dx[min(i+1, N-1)] - h_dx[max(i-1, 0)])

    tprime = np.zeros([N, 1])
    for i in range(N):
        Tgrad = utils.finite_diff(
            lambda h_: self.t1(h_), [h[i]], eps=10**-6)
        tprime[i] = Tgrad*h_dx[i]

    grad = 2.0*(q - t).T.dot(W).dot(-1.0*tprime)\
