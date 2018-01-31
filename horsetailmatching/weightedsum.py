import pdb
import time
import math
import copy
import warnings

import numpy as np

from hm import HorsetailMatching

class WeightedSum(HorsetailMatching):
    '''Class for using weighted sum of moments within an optimization.

    The code is written such that all arguments that can be used at the
    initialization of a WeightedSum object can also be set as
    attributes after creation to achieve exactly the same effect.

    :param function fqoi: function that returns the quantity of interest, it
        must take two ordered arguments - the value of the design variable
        vector and the value of the uncertainty vector.

    :param list prob_uncertainties: list of probabilistic uncertainties.
        Is a list of UncertainParameter objects, or a list of
        functions that return samples of the each uncertainty.

    :param bool/function jac: Argument that
        specifies how to evaluate the gradient of the quantity of interest.
        If False no gradients are propagated, if True the fqoi should return
        a second argument g such that g_i = dq/dx_i. If a function, it should
        have the same signature as fqoi but return g. [default False]

    :param int samples_prob: number of samples to take from the
        probabilsitic uncertainties. [default 1000]

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

    def __init__(self, fqoi, prob_uncertainties, jac=False, samples_prob=1000,
            surrogate=None, surrogate_points=None, surrogate_jac=False,
            reuse_samples=True, verbose=False,
            w1=1, w2=1):

        self.fqoi = fqoi
        self.prob_uncertainties = prob_uncertainties
        self.int_uncertainties = []
        self.jac = jac
        self.samples_prob = samples_prob
        self.reuse_samples = reuse_samples
        self.u_samples = None
        self.surrogate = surrogate
        self.surrogate_points = surrogate_points
        self.surrogate_jac = surrogate_jac
        self.verbose = verbose
        self.w1 = w1
        self.w2 = w2


##############################################################################
##  Public Methods
##############################################################################

    def evalMetric(self, x, w1=None, w2=None):
        '''Evaluates the weighted sum metric at given values of the
        design variables.

        :param iterable x: values of the design variables, this is passed as
            the first argument to the function fqoi

        :param float w1: value to weight the mean by

        :param float w2: value to weight the std by

        :return: metric_value - value of the metric evaluated at the design
            point given by x

        :rtype: float

        '''
        if w1 is None:
            w1 = self.w1
        if w2 is None:
            w2 = self.w2

        if self.verbose:
            print('----------')
            print('At design: ' + str(x))

        # Make sure dimensions are correct
        u_sample_dimensions = self._processDimensions()

        self._N_dv = len(_makeIter(x))

        if self.verbose:
            print('Evaluating surrogate')
        if self.surrogate is None:
            def fqoi(u):
                return self.fqoi(x, u)

            def fgrad(u):
                return self.jac(x, u)
            jac = self.jac
        else:
            fqoi, fgrad, surr_jac = self._makeSurrogates(x)
            jac = surr_jac

        u_samples = self._getParameterSamples(u_sample_dimensions)

        if self.verbose: print('Evaluating quantity of interest at samples')
        q_samples, grad_samples = self._evalSamples(u_samples, fqoi, fgrad, jac)

        if self.verbose: print('Evaluating metric')
        return self._evalWeightedSumMetric(q_samples, grad_samples)

##############################################################################
##  Private methods  ##
##############################################################################

    def _evalWeightedSumMetric(self, q_samples, grad_samples=None):

        fjs = np.array(q_samples).flatten()
        M = self.samples_prob

        mean = (1./M)*np.sum(fjs)
        var = (1./M)*np.sum([(fj - mean)**2 for fj in fjs])

        ws = self.w1*mean + self.w2*np.sqrt(var)

        if grad_samples is None:
            return ws
        else:
            ndv = grad_samples.shape[2]
            gradjs = grad_samples[0, :, :]

            gradient = np.zeros(ndv)
            for kdv in range(ndv):

                meang, varg = 0., 0.
                for j, fj in enumerate(fjs):
                    meang += (1./M)*float(gradjs[j, kdv])
                    varg += (1./M)*2*(fj - mean)*float(gradjs[j, kdv])

                gradient[kdv] = meang + 0.5*(var**-0.5)*varg

            return ws, gradient

    def getHorsetail(self):
        return ([0], [0]), ([0], [0]), [([0], [0])]

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
