import numpy as np
import pdb
import warnings
import random
import math

from operator import xor
from inspect import isfunction


class UncertainParameter(object):
    '''Class for handling uncertain parameters in optimization under
    uncertainty problems using horsetail matching.

    :param str distribution: distribution type of the uncertain parameter.
        Supported distributions are uniform, gaussian, custom (must provide a
        function to the pdf argument) [default uniform]

    :param double mean: mean value of the distribution [default 0]

    :param double standard_deviation: standard deviation of the distribution
        [default 1]

    :param double lower_bound: lower bound of the distribution (overrides by
        mean and standard_devaition inputs) [default -1]

    :param double upper_bound: upper bound of the distribution (overrides by
        mean and standard_devaition inputs) [default 1]

    :param function pdf: pdf function to use distribution (overrides all other
        inputs)

    *Example Declaration* ::

        >>> u = UncertainParameter('uniform')
        >>> u = UncertainParameter('gaussian')
        >>> u = UncertainParameter('uniform', lower_bound=-1, upper_bound=1)
        >>> u = UncertainParameter('gaussian', mean=0, standard_deviation=1)
        >>> u = UncertainParameter('interval', lower_bound=-1, upper_bound=1)
        >>> def myPDF(q): return 1/(2.5 - 1.5)
        >>> u = UncertainParameter('custom', pdf=myPDF, lower_bound=1.5,
                upper_bound=2.5)

    '''

    featured_dists = ['interval', 'uniform', 'gaussian', 'custom']
    default_mean = 0
    default_std = 1
    default_lb = -1
    default_ub = 1

    def __init__(self, distribution='uniform', mean=default_mean,
            standard_deviation=default_std, pdf=None,
            lower_bound=default_lb, upper_bound=default_ub):

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.pdf = pdf

        self.mean, self.standard_deviation = mean, standard_deviation
        self.lower_bound, self.upper_bound = lower_bound, upper_bound

        self.distribution = distribution

        # Private attributes
        self._returned_limits = []
        self._max_pdf_val = None

###############################################################################
## Properties with non-trivail setting behaviour
###############################################################################

    @property
    def distribution(self):
        return self._distribution

    @distribution.setter
    def distribution(self, value):

        value = value.lower()

        if value not in self.featured_dists:
            raise ValueError(
                    '''Unsupported distribution type; the following
                    distributions are currently supported: ''' +
                    ', '.join([str(e) for e in self.featured_dists]))

        if value == 'uniform':
            self.mean = 0.5*(self.lower_bound + self.upper_bound)
            self.standard_deviation = ((self.upper_bound - self.lower_bound) /
                np.sqrt(3))/2.

        elif value == 'gaussian':
            self.lower_bound = self.mean - 5*self.standard_deviation
            self.upper_bound = self.mean + 5*self.standard_deviation

        if xor(value == 'custom', self.pdf is not None):
            raise ValueError('''A pdf function is only compatible with
                custom distribution''')

        self._distribution = value

    @property
    def lower_bound(self):
        return self._lb

    @lower_bound.setter
    def lower_bound(self, value):
        if hasattr(self, '_ub') and value > self.upper_bound:
            raise ValueError('Lower bound cannot be greater than upper bound')
        self._lb = value

    @property
    def upper_bound(self):
        return self._ub

    @upper_bound.setter
    def upper_bound(self, value):
        if hasattr(self, '_lb') and value < self.lower_bound:
            raise ValueError('Lower bound cannot be greater than upper bound')
        self._ub = value


###############################################################################
## Public Methods
###############################################################################

    def getSample(self):
        '''Returns a random sample of the uncertain variable according to its
        distribution.

        *Example Usage* ::

            >>> u = UncertainParameter('uniform')
            >>> u_sample = u.getSample()

        '''

        if self.distribution == 'interval':
            if len(self._returned_limits) < 1:
                self._returned_limits.append(self.lower_bound)
                return self.lower_bound
            elif len(self._returned_limits) < 2:
                self._returned_limits.append(self.upper_bound)
                return self.upper_bound

        if self.distribution == 'uniform' or self.distribution == 'interval':
            return random.uniform(self.lower_bound, self.upper_bound)
        elif self.distribution == 'gaussian':
            return random.gauss(self.mean, self.standard_deviation)
        else:  # Rejection sampling
            if self._max_pdf_val is None:
                self._max_pdf_val = _getMaxPDFVal(self.evalPDF,
                        self.lower_bound, self.upper_bound)
            while True:
                zscale = self._max_pdf_val*1.1
                uval = (self.lower_bound +
                    np.random.random()*(self.upper_bound-self.lower_bound))
                zval = zscale*np.random.random()
                if zval < self.evalPDF(uval):
                    return uval

    def evalPDF(self, u_values):
        '''Returns the PDF of the uncertain parameter evaluated at the values
        provided in u_values.

        :param iterable u_values: values of the uncertain parameter at which to
            evaluate the PDF

        *Example Usage* ::

            >>> u = UncertainParameter('uniform')
            >>> X = numpy.linspace(-1, 1, 100)
            >>> Y = [u.evalPDF(x) for x in X]

        '''

        if isinstance(u_values, np.ndarray):
            return self._evalPDF(u_values)
        else:
            try:
                iter(u_values)
                return [self._evalPDF(u) for u in u_values]
            except:
                return self._evalPDF(u_values)

###############################################################################
## Private Methods
###############################################################################

    def _evalPDF(self, u):
        if u < self.lower_bound or u > self.upper_bound:
            return 0

        if self.distribution == 'custom':
            return self.pdf(u)

        elif self.distribution == 'uniform':
            return 1./(self.upper_bound - self.lower_bound)

        elif self.distribution == 'gaussian':
            truncconst = (_normCDF((self.upper_bound - self.mean)/
                                   self.standard_deviation) -
                          _normCDF((self.lower_bound - self.mean)/
                                   self.standard_deviation))
            return (1./truncconst)*_normPDF((u -
                self.mean)/self.standard_deviation)

        elif self.distribution == 'interval':
#            warnings.warn('Interval uncertainties have no PDF, using uniform
#                    distribution to sample')
            return 1./(self.upper_bound - self.lower_bound)

###############################################################################
## Private Functions
###############################################################################

def _normCDF(x):
    return (1. + math.erf(x / math.sqrt(2.)))/2.

def _normPDF(x):
    return (1./math.sqrt(2.*math.pi))*math.exp(-0.5*x**2)

def _getMaxPDFVal(evalPDF, lower_bound, upper_bound):
    max_pdf_val = 0
    for ui in np.linspace(lower_bound, upper_bound, 20):
        pdfi = evalPDF(ui)
        if pdfi > max_pdf_val:
            max_pdf_val = pdfi
    return max_pdf_val
