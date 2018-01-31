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

    :param list prob_uncertainties: list of probabilistic uncertainties.
        Is a list of UncertainParameter objects, or a list of
        functions that return samples of the each uncertainty.

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

    def __init__(self, fqoi, prob_uncertainties, ftarget=None, jac=False,
            samples_prob=1000, integration_points=None, kernel_bandwidth=None,
            surrogate=None, surrogate_points=None, surrogate_jac=False,
            reuse_samples=True, verbose=False):

        self.fqoi = fqoi
        self.prob_uncertainties = prob_uncertainties
        self.int_uncertainties = []
        self.ftarget = ftarget
        self.jac = jac
        self.samples_prob = samples_prob
        self.samples_int = 1
        self.integration_points = integration_points
        self.kernel_bandwidth = kernel_bandwidth
        self.reuse_samples = reuse_samples
        self.u_samples = None
        self.surrogate = surrogate
        self.surrogate_points = surrogate_points
        self.surrogate_jac = surrogate_jac
        self.verbose = verbose

        # Note that this class makes heavy use of the HorsetailMatching parent 
        # class's methods


##############################################################################
##  Public Methods
##############################################################################
    def evalMetric(self, x, method=None):
        '''Evaluates the density matching metric at a given design point.

        :param iterable x: values of the design variables, this is passed as
            the first argument to the function fqoi

        :return: metric_value - value of the metric evaluated at the design
            point given by x

        :rtype: float

        *Example Usage*::

            >>> def myFunc(x, u): return x[0]*x[1] + u
            >>> u1 = UniformParameter()
            >>> theDM = DensityMatching(myFunc, u)
            >>> x0 = [1, 2]
            >>> theDM.evalMetric(x0)

        '''
        return super(DensityMatching, self).evalMetric(x, method)

    def evalMetricFromSamples(self, q_samples, grad_samples=None, method=None):
        '''Evaluates the density matching metric from given samples of the quantity
        of interest and gradient instead of evaluating them at a design.

        :param np.ndarray q_samples: samples of the quantity of interest,
            size (M_int, M_prob)
        :param np.ndarray grad_samples: samples of the gradien,
            size (M_int, M_prob, n_x)

        :return: metric_value - value of the metric

        :rtype: float

        '''
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
