import numpy as np
import pdb
import warnings
import random
import math

from operator import xor
from inspect import isfunction


class UncertainParameter(object):
    '''Base Class for handling uncertain parameters in optimization under
    uncertainty problems using horsetail matching. If this class is used, a
    custom distribution must be provided. Otherwise one of the child classes
    UniformParameter, IntervalParameter, or GaussianParameter should be used.

    All child classes use the methods getSample and evalPDF.

    :param function pdf: pdf function of distribution. Bounds on the
        distribution should also be provided via the lower_bound and
        upper_bound arguments.

    :param double lower_bound: lower bound of the distribution [default -1]

    :param double upper_bound: upper bound of the distribution [default 1]

    *Example Declaration* ::

        >>> def myPDF(q):
            if q > 2.5 or q < 1.5:
                return 0
            else:
                return 1/(2.5 - 1.5)
        >>> u = UncertainParameter(pdf=myPDF, lower_bound=1.5, upper_bound=2.5)

    '''

    default_lb = -1
    default_ub = 1

    def __init__(self, pdf=None, lower_bound=default_lb, upper_bound=default_ub):

        self.pdf = pdf
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_interval_uncertainty = False
        self._max_pdf_val = None

###############################################################################
## Properties with non-trivail setting behaviour
###############################################################################

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

            >>> u = UniformParameter()
            >>> u_sample = u.getSample()

        '''
        ## _getSample is overwritten in child classes
        return self._getSample()

    def evalPDF(self, u_values):
        '''Returns the PDF of the uncertain parameter evaluated at the values
        provided in u_values.

        :param iterable u_values: values of the uncertain parameter at which to
            evaluate the PDF

        *Example Usage* ::

            >>> u = UniformParameter()
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

    def _getSample(self):
        if self._max_pdf_val is None:
            self._max_pdf_val = self._getMaxPDFVal(self.evalPDF,
                    self.lower_bound, self.upper_bound)
        while True:
            zscale = self._max_pdf_val*1.1
            uval = (self.lower_bound +
                np.random.random()*(self.upper_bound-self.lower_bound))
            zval = zscale*np.random.random()
            if zval < self.evalPDF(uval):
                return uval

    def _getMaxPDFVal(self, evalPDF, lower_bound, upper_bound):
        max_pdf_val = 0
        for ui in np.linspace(lower_bound, upper_bound, 20):
            pdfi = evalPDF(ui)
            if pdfi > max_pdf_val:
                max_pdf_val = pdfi
        return max_pdf_val

    def _evalPDF(self, u):
        if u < self.lower_bound or u > self.upper_bound:
            return 0
        else:
            return self.pdf(u)

###############################################################################
## Child classes for specific distributions
## They should override the __init__, _getSample and _evalPDF methods
###############################################################################

class UniformParameter(UncertainParameter):
    '''Class for creating uniform uncertain parameters for use with
    horsetail matching.

    :param double lower_bound: lower bound of the distribution [default -1]

    :param double upper_bound: upper bound of the distribution [default 1]

    *Example Declaration* ::

        >>> u = UniformParameter()
        >>> u = UniformParameter(lower_bound=-2, upper_bound=2)

    '''

    default_lb = -1
    default_ub = 1

    def __init__(self, lower_bound=default_lb, upper_bound=default_ub):

        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        self.is_interval_uncertainty = False

    def _getSample(self):
        return random.uniform(self.lower_bound, self.upper_bound)

    def _evalPDF(self, u):
        if u < self.lower_bound or u > self.upper_bound:
            return 0
        else:
            return 1./(self.upper_bound - self.lower_bound)

class IntervalParameter(UniformParameter):
    '''Class for creating interval uncertain parameters for use with
    horsetail matching.

    :param double lower_bound: lower bound of the interval [default -1]

    :param double upper_bound: upper bound of the interval [default 1]

    *Example Declaration* ::

        >>> u = IntervalParameter()
        >>> u = IntervalParameter(lower_bound=-2, upper_bound=2)

    '''

    default_lb = -1
    default_ub = 1

    def __init__(self, lower_bound=default_lb, upper_bound=default_ub):

        super(IntervalParameter, self).__init__(lower_bound, upper_bound)
        self.is_interval_uncertainty = True
        self._returned_limits = []

    def _getSample(self):

        if len(self._returned_limits) < 1:
            self._returned_limits.append(self.lower_bound)
            return self.lower_bound
        elif len(self._returned_limits) < 2:
            self._returned_limits.append(self.upper_bound)
            return self.upper_bound
        else:
            return random.uniform(self.lower_bound, self.upper_bound)

class GaussianParameter(UncertainParameter):
    '''Class for creating gaussian uncertain parameters for use with
    horsetail matching.

    :param double mean: mean value of the distribution. [default 0]

    :param double standard_deviation: standard deviation of the distribution.
        [default 1]

    *Example Declaration* ::

        >>> u = GaussianParameter()
        >>> u = GaussianParameter(mean=1, standard_deviation=2)

    '''

    default_mean = 0
    default_std = 1

    def __init__(self, mean=default_mean, standard_deviation=default_std):

        self.mean = mean
        self.standard_deviation = standard_deviation
        self.is_interval_uncertainty = False

    def _getSample(self):
        return random.gauss(self.mean, self.standard_deviation)

    def _evalPDF(self, u):
        return self._normPDF((u - self.mean)/self.standard_deviation)

    def _normPDF(self, x):
        return (1./math.sqrt(2.*math.pi))*math.exp(-0.5*x**2)
