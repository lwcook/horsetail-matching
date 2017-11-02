import pdb
import time
import math
import copy
import warnings

import numpy as np

from hm import HorsetailMatching

class DensityMatching(HorsetailMatching):
    '''Class for using density matching within an optimization. The main
    functionality is to evaluate the density matching
    metric (and optionally its gradient) that can be used with external
    optimizers. 

    The code is written such that all arguments that can be used at the
    initialization of a DensityMatching object can also be set as
    attributes after creation to achieve exactly the same effect.

    :param function fqoi: function that returns the quantity of interest, it
        must take two ordered arguments - the value of the design variable
        vector and the value of the uncertainty vector.

    :param list uncertain_parameters: list of UncertainParameter objects
        that describe the uncertain inputs for the problem (they must have
        the getSample() method).

    :param function ftarget: function that returns the value of the target
        PDF function.

    :param bool/function jac: Argument that
        specifies how to evaluate the gradient of the quantity of interest.
        If False no gradients are propagated, if True the fqoi should return
        a second argument g such that g_i = dq/dx_i. If a function, it should
        have the same signature as fqoi but return g. [default False]

    :param int samples_prob: number of samples to take from the
        probabilsitic uncertainties. [default 1000]

    :param list integration_points:
        The integration point values to use when evaluating the metric using
        kernels [by default 100 points spread over 3 times the range of
        the samples of q obtained the first time the metric is evaluated]

    :param number kernel_bandwidth: The bandwidth
        used in the kernel function [by default it is found the first time
        the metric is evaluated using Scott's rule]

    :param function surrogate: Surrogate that is created at every design
        point to be sampled instead of fqoi. It should be a function that
        takes two arguments - an array with values of the uncertainties at
        which to fit the surrogate of size (num_quadrature_points,
        num_uncertainties), and an array of quantity of interest values
        corresponding to these uncertainty values to which to fit the surrogate
        of size (num_quadrature_points). It should return a functio that
        predicts the qoi at an aribtrary value of the uncertainties.
        [default None]

    :param list surrogate_points: Only with a surrogate. List of points at
        which fqoi is evaluated to give values to fit the surrogates to. These
        are passed to the surrogate function along with the qoi evaluated at
        these points when the surrogate is fitted [by default tensor
        quadrature of 5 points in each uncertain dimension is used]

    :param bool/function surrogate_jac: Only with a surrogate.  Specifies how
        to take surrogates of the gradient. It works similarly to the
        jac argument: if False, the same surrogate is fitted to fqoi and each
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

    '''

    def __init__(self, fqoi, uncertain_parameters, ftarget=None, jac=False,
            samples_prob=1000, integration_points=None, kernel_bandwidth=None,
            surrogate=None, surrogate_points=None, surrogate_jac=False,
            reuse_samples=True, verbose=False):

        self.fqoi = fqoi
        self.uncertain_parameters = uncertain_parameters
        self.ftarget = ftarget
        self.jac = jac
        self.samples_prob = samples_prob
        self.integration_points = integration_points
        self.kernel_bandwidth = kernel_bandwidth
        self.reuse_samples = reuse_samples
        self.u_samples = None
        self.surrogate = surrogate
        self.surrogate_points = surrogate_points
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
        self._u_params = _makeIter(params)
        if len(self._u_params) == 0:
            raise ValueError('No uncertain parameters provided')

        self._u_int = []
        self._u_prob = []
        for ii, u in enumerate(self._u_params):
            self._u_prob.append((ii, u))

    @property
    def u_samples(self):
        return self._u_samples

    @u_samples.setter
    def u_samples(self, samples):
        if samples is not None:
            if (not isinstance(samples, np.ndarray) or
                    samples.shape != self._processDimensions()):
                raise TypeError('u_samples should be a np.array of size'
                        '(samples_int, samples_prob, num_uncertanities)')
        self._u_samples = samples


##############################################################################
##  Public Methods
##############################################################################

    def evalMetric(self, x, wide=False):
        '''Evaluates the density matching metric at given values of the
        design variables.

        :param iterable x: values of the design variables, this is passed as
            the first argument to the function fqoi

        :return: metric_value - value of the metric evaluated at the design
            point given by x

        :rtype: float

        '''

        if self.verbose:
            print('----------')
            print('At design: ' + str(x))

        # Make sure dimensions are correct
        u_sample_dimensions = self._processDimensions()

        self._N_dv = len(_makeIter(x))

        if self.verbose:
            print('Evaluating surrogate')
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
        return self._evalDensityMetric(q_samples, grad_samples)

    def getPDF(self):
        '''Function that gets vectors of the pdf and target at the last design
        evaluated.

        :return: tuple of q values, pdf values, target values
        '''

        if hasattr(self, '_qplot'):

            return self._qplot, self._hplot, self._tplot

        else:
            raise ValueError('''The metric has not been evaluated at any
                    design point so the PDF cannot get obtained''')

##############################################################################
##  Private methods  ##
##############################################################################

    def _evalDensityMetric(self, q_samples, grad_samples=None):
        
        if self.integration_points is None:
            q_min = np.amin(q_samples)
            q_max = np.amax(q_samples)
            q_range = q_max - q_min
            fis = np.linspace(q_min - q_range, q_max + q_range, 1000)
            self.integration_points = fis
        else:
            fis = self.integration_points

        # If kernel bandwidth not specified, find it using Scott's rule
        if self.kernel_bandwidth is None:
            if abs(np.max(q_samples) - np.min(q_samples)) < 1e-6:
                bw = 1e-6
            else:
                bw = ((4/(3.*q_samples.shape[1]))**(1/5.)
                          *np.std(q_samples[0,:]))
            self.kernel_bandwidth = bw
        else:
            bw = self.kernel_bandwidth

        fjs = np.array(q_samples)

        N = len(fis)
        M = self.samples_prob

        t = np.array([float(self.ftarget(fi)) for fi in fis]).reshape([N, 1])

        # column vector - row vector to give matrix
        delf = fis.reshape([N, 1]) - fjs.reshape([1, M])

        const_term = 1.0/(M * np.sqrt(2*np.pi*bw**2))
        K = const_term * np.exp((-1./2.) * (delf/bw)**2)

        Ks = np.dot(K, np.ones([M, 1])).reshape([N, 1])

        W = np.zeros([N, N])     # Trapezium rule weighting matrix
        for i in range(N):
            W[i, i] = (fis[min(i+1, N-1)] - fis[max(i-1, 0)])*0.5

        l2norm = float((t - Ks).T.dot(W.dot((t - Ks))))

        self._qplot = fis
        self._hplot = Ks
        self._tplot = t


        if grad_samples is None:
            return l2norm
        else:
            ndv = grad_samples.shape[2]
            gradjs = grad_samples[0, :, :]
            Kprime = const_term * np.exp((-1./2.) * (delf/bw)**2) *\
                        delf / bw**2 * -1.

            Fprime = np.zeros([M, ndv])
            for kdv in range(ndv):
                Fprime[:, kdv] = gradjs[:, kdv]

            gradient = 2*(t - Ks).T.dot(W.dot(Kprime.dot(Fprime))).reshape(ndv)

            return l2norm, gradient


    def _processDimensions(self):

        N_u = len(self._u_prob)
        u_sample_dim = (1, self.samples_prob, N_u)
        self.samples_int = 1

        return u_sample_dim

## Private utility functions

#def _finDiff(fobj, dv, f0=None, eps=10**-6):
#
#    if f0 is None:
#        f0 = fobj(dv)
#
#    fbase = copy.copy(f0)
#    fnew = fobj(dv + eps)
#    return float((fnew - fbase)/eps)

def _makeIter(x):
    try:
        iter(x)
        return [xi for xi in x]
    except:
        return [x]
