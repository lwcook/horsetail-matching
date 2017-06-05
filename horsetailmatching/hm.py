import pdb
import time
import warnings
from collections import Iterable

import numpy as np
import scipy.special as scp
import matplotlib.pyplot as plt

import utilities as utils


class HorsetailMatching(object):
    '''Class for using horsetail matching within an optimization. The main
    functionality is to evaluate the horsetail matching
    metric (and optionally its gradient) that can be used with external
    optimizers.

    :param function fqoi: function that returns the quantity of interest, it
        must take two ordered arguments - the value of the design variable
        vector and the value of the uncertainty vector.

    :param list uncertain_parameters: list of Parameter objects that
        describe the uncertain inputs for the problem. They must have the
        getSample() method.

    :param bool/function jac: Only for method = 'kernel'. Argument that
        specifies how to evaluate the gradient of the quantity of interest.
        If False no gradients are used, if True the fqoi should return a
        second argument g such that g_i = dq/dx_i. If a function, it should
        have the same signature as fqoi but return g. [default False]

    :param str method: method with which to evaluate the horsetil matching
        metric, can be 'empirical' or 'kernel' [default 'empirical' if
        jac is False else default 'kernel'].

    :param function ftarget: function that returns the value of the target
        inverse CDF given a value in [0,1]. Can be a tuple that gives two
        target fuctions, one for the upper bound and one for the lower bound on
        the CDF under mixed uncertainties [default t(h) = 0]

    :param int n_samples_prob: number of samples to take from the
        probabilsitic uncertainties. [default 500]

    :param int n_samples_int: number of samples to take from the
        interval uncertainties. Note that under mixed uncertainties, a nested
        loop is used to evaluate the metric so the total number of
        samples will be n_samples_prob*n_samples_int. [default 20]

    :param list q_integration_points: Only for method='kernel'.
        The integration points to use when evaluating the metric using
        kernels [by default 100 points spread over 3 times the range of
        the samples of q obtained the first time the metric is evaluated]

    :param number kernel_bandwidth: Only for method='kernel'. The bandwidth
        used in the kernel function [by default is found the first time the
        metric is evaluated using Scott's rule]

    :param function surrogate: Surrogate that is created at every design
        point to be sampled instead of fqoi. It should be a function that
        takes two arguments - an array with values of the uncertainties at
        which to fit the surrogate of size (num_quadrature_points,
        num_uncertainties), and an array of quantity of interest values
        corresponding to these uncertainty values to which to fit the surrogate
        of size (num_quadrature_points) [default None]

    :param list u_surrogate_points: Only with a surrogate. List of points at
        which fqoi is evaluated to give values to fit the surrogates to. These
        are passed to the surrogate function along with the qoi evaluated at
        these points when the surrogate is fitted [by default tensor
        quadrature of 5 points in each uncertain dimension is used]

    :param bool/function surrogate_jac: Only with a surrogate.  Specifies how
        to take surrogates of the gradient. It works similarly to the
        jac argument: if False, the same surrgate is fitted to fqoi and each
        component of its gradient, if True, the surrogate function is
        expected to take a third argument - an array that is the gradient
        at each of the quadrature points of size
        (num_quadrature_points, num_design_variables). If a function, then
        instead the array of uncertainty values and the array of gradient
        values are passed to this function and it should return a function for
        the surrogate model of the gradient.

    :param bool reuse_samples: If True will reuse the same set of samples of
        the uncertainties for evaluating the metric at any value of the
        design variables, if False wise will re-sample every time evalMetric
        is called [default True]

    :param bool verbose: If True will print out details [default False].

    *Example Declarations*::

        >>> from horsetailmatching import HorsetailMatching,
                UncertainParameter, PolySurrogate

        >>> def myFunc(x, u): return x[0]*x[1] + u
        >>> def myGrad(x, u): return [x[1], x[0]]
        >>> def myTarg1(h): return 1-h**3
        >>> def myTarg2(h): return 2-h**3

        >>> u1 = UncertainParameter('uniform')
        >>> u2 = UncertainParameter('interval')
        >>> poly = PolySurrogate(dimensions=2)
        >>> poly_points = poly.getQuadraturePoints()

        >>> theHM = HorsetailMatching(myFunc, u)
        >>> theHM = HorsetailMatching(myFunc, u, jac=myGrad, method='kernel')
        >>> theHM = HorsetailMatching(myFunc, u, ftarget=myTarg1)
        >>> theHM = HorsetailMatching(myFunc, u, ftarget=(myTarg1, myTarg2))
        >>> theHM = HorsetailMatching(myFunc, u, n_samples_prob=500,
                n_samples_int = 50)
        >>> theHM = HorsetailMatching(myFunc, u, method='kernel',
                q_integration_points=numpy.linspace(0, 10, 100),
                kernel_bandwidth=0.01)
        >>> theHM = HorsetailMatching(myFunc, u,
                surrogate=poly.surrogate, surrogate_jac=False,
                u_surrogate_points=poly_points)
        >>> theHM = HorsetailMatching(myFunc, u, verbose=True,
                reuse_samples=True)

    '''

    def __init__(self, fqoi, uncertain_parameters, jac=False, method=None,
            ftarget=None,
            n_samples_prob=500, n_samples_int=20, q_integration_points=None,
            kernel_bandwidth=None, alpha=500,
            surrogate=None, u_surrogate_points=None, surrogate_jac=False,
            reuse_samples=True, verbose=False):

        self.fqoi = fqoi
        self.uncertain_parameters = uncertain_parameters
        self.jac = jac
        self.method = method # Must be done after setting jac
        self.ftarget = ftarget
        self.n_samples_prob = n_samples_prob
        self.n_samples_int = n_samples_int
        self.q_integration_points = q_integration_points
        self.kernel_bandwidth = kernel_bandwidth
        self.alpha = alpha
        self.reuse_samples = reuse_samples
        self.u_samples = None
        self.surrogate = surrogate
        self.u_surrogate_points = u_surrogate_points
        self.surrogate_jac = surrogate_jac
        self.verbose = verbose

###############################################################################
## Properties with non-trivial setting behaviour
###############################################################################

    @property
    def uncertain_parameters(self):
        return self._u_params

    @uncertain_parameters.setter
    def uncertain_parameters(self, params):
        self._u_params = utils.makeIter(params)
        if len(self._u_params) == 0:
            raise ValueError('No uncertain parameters provided')

        self._u_int, self._u_prob = [], []
        for ii, u in enumerate(self._u_params):
            if u.distribution == 'interval':
                self._u_int.append((ii, u))
            else:
                self._u_prob.append((ii, u))

    @property
    def method(self):
        return self._method

    @method.setter
    def method(self, value):
        if value is None:
            if self.jac is False:
                self._method = 'empirical'
            else:
                self._method = 'kernel'
        else:
            self._method = value

    @property
    def ftarget(self):
        return self._ftarget

    @ftarget.setter
    def ftarget(self, value):
        def standardTarget(h):
            return 0
        try:
            iter(value)
            self._ftarg_u = value[0]
            self._ftarg_l = value[1]
            self._ftarget = value
        except:
            if value is None:
                self._ftarget = standardTarget
            else:
                self._ftarget = value
            self._ftarg_u = self._ftarget
            self._ftarg_l = self._ftarget

    @property
    def u_samples(self):
        return self._u_samples

    @u_samples.setter
    def u_samples(self, samples):
        if samples is not None:
            if not (isinstance(samples, np.ndarray) or
                    samples.size != self._processDimensions()):
                raise TypeError('u_samples should be a np.array of size'
                        '(n_samples_prob, n_samples_int, num_uncertanities)')
        self._u_samples = samples

##############################################################################
##  Public Methods
##############################################################################

    def evalMetric(self, x, method=None):
        '''Evaluates the horsetail matching metric at given values of the
        design variables.

        :param iterable x: values of the design variables, this is passed as
            the first argument to the function, fqoi, provided to evaluate the
            quantity of intetest
        :param str method: method to use to evaluate the metric ('empirical' or
            'kernel')

        :return: metric_value - value of the metric evaluated at the design
            point given by x

        :rtype: float

        *Example Usage*::

            >>> def myFunc(x, u): return x[0]*x[1] + u
            >>> u = UncertainParameter('uniform')
            >>> theHM = HorsetailMatching(myFunc, u)
            >>> x0 = [1, 2]
            >>> theHM.evalMetric(x0)

        '''
        if self.verbose:
            print('----------')
            print('At design: ' + str(x))

        # Make sure dimensions are correct
        u_sample_dimensions = self._processDimensions()

        self._N_dv = len(utils.makeIter(x))

        if method is None:
            method = self.method

        if self.verbose: print('Evaluating surrogate')
        if self.surrogate is None:
            def fqoi(u): return self.fqoi(x, u)
            def fgrad(u): return self.jac(x, u)
            jac = self.jac
        else:
            fqoi, fgrad, surr_jac = self._makeSurrogates(x)
            jac = surr_jac

        u_samples = self._getParameterSamples(u_sample_dimensions)

        if self.verbose: print('Evaluating quantity of interest at samples')
        q_samples, grad_samples = self._evalSamples(u_samples, fqoi, fgrad, jac)

        if self.verbose: print('Evaluating metric')
        if method.lower() == 'empirical':
            if self.jac:
                raise TypeError( 'Empicial method does not support gradients')
            else:
                return self._evalMetricEmpirical(q_samples)
        elif method.lower() == 'kernel':
            if self.kernel_bandwidth is None:
                self.kernel_bandwidth = ((4/(3.*q_samples.shape[1]))**(1/5.) *
                        np.std(q_samples[0,:]))
            return self._evalMetricKernel(q_samples, grad_samples)
        else:
            raise ValueError('Unsupported metric evalation method')

    def plotHorsetail(self, *plotargs, **plotkwargs):
        '''Function that plots the horsetail of the last design at which the
        evalMetric method was called. It uses the matplotlib module.
        Any arguments and key word aguments are passed to the plot() function
        used to plot the two horsetail curves.

        *Example Usage*::

            >>> def myFunc(x, u): return x[0]*x[1] + u
            >>> u = UncertainParameter('uniform')
            >>> theHM = HorsetailMatching(myFunc, u)
            >>> theHM.plotHorsetail('b', lw=0.5)
            >>> matplotlib.pyplot.show()

        '''

        ql, qu, hl, hu = self._ql, self._qu, self._hl, self._hu
        qh, hh = self._qh, self._hh

        if self.q_integration_points is not None:
            ql, hl = _appendPlotArrays(ql, hl, self.q_integration_points)
            qu, hu = _appendPlotArrays(qu, hu, self.q_integration_points)

        for qi, hi in zip(qh, hh):
            plt.plot(qi, hi, c='grey', alpha=0.5, lw=0.5)
        plt.plot(ql, hl, *plotargs, **plotkwargs)
        plt.plot([self._ftarg_l(hi) for hi in hl], hl, 'k:')
        plt.plot(qu, hu, *plotargs, **plotkwargs)
        plt.plot([self._ftarg_u(hi) for hi in hu], hu, 'k:')
        plt.ylim([0,1])
        plt.xlabel('q - Quantity of Interest')
        plt.ylabel('h - CDF')

##############################################################################
##  Private methods  ##
##############################################################################

    def _evalMetricEmpirical(self, q_samples):

        h_htail = np.zeros(q_samples.shape)
        q_htail = np.zeros(q_samples.shape)
        for ii in np.arange(q_samples.shape[0]):
            # Get empirical CDF by sorting samples at each value of intervals
            q_htail[ii, :] = np.sort(q_samples[ii, :])
            M = q_samples.shape[1]
            h_htail[ii, :] = [(1./M)*(0.5 + j) for j in range(M)]

        if q_samples.shape[0] > 1:
            q_u = q_htail.min(axis=0)
            q_l = q_htail.max(axis=0)
        else:
            q_u, q_l = q_htail[0], q_htail[0]
        h_u, h_l = h_htail[0], h_htail[0]  # h is same for all ECDFs

        D_u, D_l = 0., 0.
        for (qui, hui), (qli, hli) in zip(zip(q_u, h_u), zip(q_l, h_l)):
            D_u += (1./self.n_samples_prob)*(qui - self._ftarg_u(hui))**2
            D_l += (1./self.n_samples_prob)*(qli - self._ftarg_l(hli))**2

        dhat = np.sqrt(D_u + D_l)
        self._ql, self._qu, self._hl, self._hu = q_l, q_u, h_l, h_u
        self._qh, self._hh = q_htail, h_htail
        return dhat

    def _evalMetricKernel(self, q_samples, grad_samples=None):

        ## Initalize arrays and prepare calculation
        if self.q_integration_points is None:
            q_min = np.amin(q_samples)
            q_max = np.amax(q_samples)
            q_range = q_max - q_min
            qis = np.linspace(q_min - q_range, q_max + q_range, 100)
        else:
            qis = self.q_integration_points
        N_quad = len(qis)
        M_prob = self.n_samples_prob
        M_int = self.n_samples_int

        fhtail = np.zeros([M_int, N_quad])
        qhtail = np.zeros([M_int, N_quad])
        hu, hl = np.zeros(N_quad), np.zeros(N_quad)

        if grad_samples is not None:
            fht_grad = np.zeros([M_int, N_quad, self._N_dv])
            fl_prime = np.zeros([N_quad, M_int])
            fu_prime = np.zeros([N_quad, M_int])
            hl_grad = np.zeros([N_quad, self._N_dv])
            hu_grad = np.zeros([N_quad, self._N_dv])
            Du_grad = np.zeros(self._N_dv)
            Dl_grad = np.zeros(self._N_dv)

        # ALGORITHM 1 from publication
        # Evaluate all individual CDFs and their gradients
        for ii in np.arange(M_int):
            qjs = q_samples[ii, :]
            rmat = qis.reshape([N_quad, 1])-qjs.reshape([1, M_prob])

            if grad_samples is not None:
                Kcdf, Kprime = _kernel(rmat, M_prob, self.kernel_bandwidth,
                        bGrad=True)
                for ix in np.arange(self._N_dv):
                    grad_js = grad_samples[ii, :, ix]
                    fht_grad[ii, :, ix] = Kprime.dot(-1*grad_js)
            else:
                Kcdf = _kernel(rmat, M_prob, self.kernel_bandwidth,
                        bGrad=False)

            fhtail[ii, :] = Kcdf.dot(np.ones([M_prob, 1])).flatten()
            qhtail[ii, :] = qis

        # ALGORITHM 2 from publication
        # Find horsetail curves - envelope of the CDFs and their gradients
        for iq in np.arange(N_quad):

            hu[iq] = _extalg(fhtail[:, iq], self.alpha)
            hl[iq] = _extalg(fhtail[:, iq], -1*self.alpha)

            if grad_samples is not None:
                fu_prime[iq, :] = _extgrad(fhtail[:, iq], self.alpha)
                fl_prime[iq, :] = _extgrad(fhtail[:, iq], -1*self.alpha)
                for ix in np.arange(self._N_dv):
                    his_grad = fht_grad[:, iq, ix]
                    hu_grad[iq, ix] = fu_prime[iq, :].dot(his_grad)
                    hl_grad[iq, ix] = fl_prime[iq, :].dot(his_grad)

        # ALGORITHM 3 from publication
        # Evaluate overall metric and gradient using matrix multipliation
        tu = np.array([self._ftarg_u(hi) for hi in hu])
        tl = np.array([self._ftarg_l(hi) for hi in hl])

        Du = _matrix_integration(qis, hu, tu)
        Dl = _matrix_integration(qis, hl, tl)
        dhat = float(np.sqrt(Du + Dl))

        self._ql, self._qu, self._hl, self._hu = qis, qis, hl, hu
        self._qh, self._hh = qhtail, fhtail
        self._Dl, self._Du = Dl, Du
        if self.verbose:
            print('Metric: ' + str(dhat))

        if grad_samples is not None:
            tu_pr = np.array([utils.finDiff(self._ftarg_u, hi) for hi in hu])
            tl_pr = np.array([utils.finDiff(self._ftarg_l, hi) for hi in hl])
            for ix in np.arange(self._N_dv):
                Du_grad[ix] = _matrix_grad(qis, hu, hu_grad[:, ix], tu, tu_pr)
                Dl_grad[ix] = _matrix_grad(qis, hl, hl_grad[:, ix], tl, tl_pr)

            dhat_grad = (0.5*(Du+Dl)**(-0.5)*(Du_grad + Dl_grad))
            if self.verbose:
                print('Gradient: ' + str([g for g in dhat_grad]))

            return dhat, dhat_grad

        else:
            return dhat

    def _makeSurrogates(self, x):

        # Get quadrature points
        if self.u_surrogate_points is None:
            N_u = len(self._u_prob) + len(self._u_int)
            mesh = np.meshgrid(*[np.linspace(-1, 1, 5) for n in np.arange(N_u)],
                    copy=False)
            u_sparse = np.vstack([m.flatten() for m in mesh]).T
        else:
            u_sparse = self.u_surrogate_points

        N_sparse = u_sparse.shape[0]
        q_sparse = np.zeros(N_sparse)

        # Get surrogates in correct form
        if not self.jac:
            for iu, u in enumerate(u_sparse):
                q_sparse[iu] = self.fqoi(x, u)

            surr_qoi = self.surrogate(u_sparse, q_sparse)

            def fqoi(u): return surr_qoi(u)
            fgrad = False
            surr_jac = False

        else:
            g_sparse = np.zeros([N_sparse, self._N_dv])
            fpartial = [lambda u: 0 for _ in np.arange(self._N_dv)]

            for iu, u in enumerate(u_sparse):
                if isinstance(self.jac, bool) and self.jac:
                    q_sparse[iu], g_sparse[iu, :] = self.fqoi(x, u)
                else:
                    q_sparse[iu] = self.fqoi(x, u)
                    g_sparse[iu, :] = self.jac(x, u)

            if not self.surrogate_jac:
                surr_qoi = self.surrogate(u_sparse, q_sparse)
                for k in np.arange(self._N_dv):
                    fpartial[k] = self.surrogate(u_sparse, g_sparse[:, k])
                def surr_grad(u): return [f(u) for f in fpartial]
            else:
                if isinstance(self.surrogate_jac, bool) and self.surrogate_jac:
                    surr_qoi, surr_grad = self.surrogate(
                                u_sparse, q_sparse, g_sparse)
                else:
                    surr_qoi  = self.surrogate(u_sparse, q_sparse)
                    surr_grad = self.surrogate_jac(u_sparse, g_sparse)

            def fqoi(u): return(surr_qoi(u))
            def fgrad(u): return(surr_grad(u))
            surr_jac = fgrad

        return fqoi, fgrad, surr_jac

    def _getParameterSamples(self, u_sample_dimensions):

        get_new = True
        if self.reuse_samples and self.u_samples is not None:
            if self.u_samples.shape != u_sample_dimensions:
                if self.verbose:
                    print('''Stored samples do not match current dimensions,
                            getting new samples''')
            else:
                get_new = False

        if get_new:
            if self.verbose:
                print('Getting uncertain parameter samples')
            N_u = len(self._u_int) + len(self._u_prob)

            u_samples = np.zeros(u_sample_dimensions)

            # Sample over interval uncertainties, and then at each sampled
            # value sample over the probabilistic uncertainties
            for ii in np.arange(u_samples.shape[0]):
                u_i = self._getOneSample(self._u_int, N_u)

                u_sub = np.zeros([u_samples.shape[1], N_u])
                for jj in np.arange(u_samples.shape[1]):
                    u_p = self._getOneSample(self._u_prob, N_u)
                    u_sub[jj,:] = u_i + u_p

                u_samples[ii,:,:] = u_sub

            self.u_samples = u_samples
            return u_samples
        else:
            if self.verbose: print('Re-using stored samples')
            return self.u_samples

    def _getOneSample(self, u_params, N_u):
        vu = np.zeros(N_u)
        if len(u_params) > 0: # Return zeros if no parameters given
            for (i, u) in u_params:
                vu[i] = (u.getSample())
        return vu

    def _evalSamples(self, u_samples, fqoi, fgrad, jac):

        # Array of shape (M_int, M_prob)
        grad_samples = None
        q_samples = np.zeros([self.n_samples_int, self.n_samples_prob])
        if not jac:
            for ii in np.arange(q_samples.shape[0]):
                for jj in np.arange(q_samples.shape[1]):
                    q_samples[ii, jj] = fqoi(u_samples[ii, jj])
        else:
            grad_samples = np.zeros([self.n_samples_int, self.n_samples_prob,
                self._N_dv])
            for ii in np.arange(q_samples.shape[0]):
                for jj in np.arange(q_samples.shape[1]):
                    if isinstance(jac, bool) and jac:
                        (q, grad) = fqoi(u_samples[ii, jj])
                        q_samples[ii, jj] = float(q)
                        grad_samples[ii, jj, :] = [_ for _ in grad]
                    else:
                        q_samples[ii, jj] = fqoi(u_samples[ii, jj])
                        grad_samples[ii, jj, :] = fgrad(u_samples[ii, jj])

        return q_samples, grad_samples

    def _processDimensions(self):

        N_u = len(self._u_int) + len(self._u_prob)

        # Mixed uncertainties
        if len(self._u_int) > 0 and len(self._u_prob) > 0:
            u_sample_dim = (self.n_samples_int, self.n_samples_prob, N_u)

        # Probabilistic uncertainties
        elif len(self._u_int) == 0:
            self.n_samples_int = 1
            u_sample_dim = (1, self.n_samples_prob, N_u)

        # Interval Uncertainties
        elif len(self._u_prob) == 0:
            self.n_samples_prob = 1
            u_sample_dim = (self.n_samples_int, 1, N_u)
            self.kernel_bandwidth = 1e-3

        return u_sample_dim

##############################################################################
##  Private functions
##############################################################################

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
    term2 = 1 + alpha*(xarr - _extalg(xarr, alpha))

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

def _matrix_grad(q, h, h_dx, t, t_prime):
    ''' Returns the gradient with respect to a single variable'''

    N = len(q)
    W = np.zeros([N, N])
    Wprime = np.zeros([N, N])
    for i in range(N):
        W[i, i] = 0.5*(h[min(i+1, N-1)] - h[max(i-1, 0)])
        Wprime[i, i] = \
            0.5*(h_dx[min(i+1, N-1)] - h_dx[max(i-1, 0)])

    tgrad = np.array([t_prime[i]*h_dx[i] for i in np.arange(N)])

    grad = 2.0*(q - t).T.dot(W).dot(-1.0*tgrad) \
            + (q - t).T.dot(Wprime).dot(q - t)

    return grad

def _appendPlotArrays(q, h, q_integration_points):
    q = np.insert(q, 0, q[0])
    h = np.insert(h, 0, 0)
    q = np.insert(q, 0, min(q_integration_points))
    h = np.insert(h, 0, 0)
    q = np.append(q, q[-1])
    h = np.append(h, 1)
    q = np.append(q, max(q_integration_points))
    h = np.append(h, 1)
    return q, h

def _appendPlotArrays(q, h, q_integration_points):
    q = np.insert(q, 0, q[0])
    h = np.insert(h, 0, 0)
    q = np.insert(q, 0, min(q_integration_points))
    h = np.insert(h, 0, 0)
    q = np.append(q, q[-1])
    h = np.append(h, 1)
    q = np.append(q, max(q_integration_points))
    h = np.append(h, 1)
    return q, h


