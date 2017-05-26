import numpy as np
import pdb
from operator import xor
import utilities as utils
from inspect import isfunction
import warnings
from scipy.stats import norm


class UncertainParameter():
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
    '''

    featured_dists = ['interval', 'uniform', 'gaussian', 'custom']
    default_mean = 0
    default_std = 1
    default_lb = -1
    default_ub = 1

    def __init__(self, distribution='uniform', mean=default_mean,
            standard_deviation=default_std, pdf=None,
            lower_bound=default_lb, upper_bound=default_ub):

        self.distribution = distribution.lower()

        self.mu, self.std = mean, standard_deviation
        self.lb, self.ub = lower_bound, upper_bound
        self.fpdf = pdf

        # Private attributes
        self._samples_found = []
        self._max_pdf_val = 0

        self._check_attributes()

    def getSample(self):
        '''Returns a random sample of the uncertain variable according to its
        distribution using rejection sampling.'''

        while True:
            zscale = self._max_pdf_val*1.1
            uval = self.lb + np.random.random()*(self.ub-self.lb)
            zval = zscale*np.random.random()
            if zval < self.evalPDF(uval):
                return uval

    def evalPDF(self, u_values):
        '''Returns the PDF of the uncertain parameter evaluated at the values
        provided in u_values.

        :param iterable u_values: values of the uncertain parameter at which to
            evaluate the PDF'''

        if isinstance(u_values, np.ndarray):
            return self._evalPDF(u_values)
        else:
            try:
                iter(u_values)
                return [self._evalPDF(u) for u in u_values]
            except:
                return self._evalPDF(u_values)

    def _evalPDF(self, u):
        if u < self.lb or u > self.ub:
            return 0

        if self.distribution == 'custom':
            return self.fpdf(u)

        elif self.distribution == 'uniform':
            return 1./(self.ub - self.lb)

        elif self.distribution == 'gaussian':
            if self.lb is not None and self.ub is not None:
                truncconst = norm.cdf((self.ub - self.mu)/self.std) - \
                    norm.cdf((self.lb - self.mu)/self.std)
            else:
                truncconst = 1
            return (1./truncconst)*norm.pdf((u - self.mu)/self.std)

        elif self.distribution == 'interval':
#            warnings.warn('Interval uncertainties have no PDF, using uniform
#                    distribution to sample')
            return 1./(self.ub - self.lb)

    def _check_attributes(self):

        if self.distribution not in self.featured_dists:
            raise ValueError(
                    '''Unsupported distribution type; the following
                    distributions are currently supported: ''' +
                    ', '.join([str(e) for e in self.featured_dists]))

        if self.ub < self.lb:
            raise ValueError('Upper bound is less than lower bound')

        if self.distribution == 'uniform':
            self.mu = 0.5*(self.lb + self.ub)
            self.std = 0.5*(self.ub - self.lb) / np.sqrt(3)

        elif self.distribution == 'gaussian':
            self.lb = self.mu - 5*self.std
            self.ub = self.mu + 5*self.std

        if xor(self.distribution == 'custom', self.fpdf is not None):
            raise ValueError('''A pdf function is only compatible with
                custom distribution''')

        for ui in np.linspace(self.lb, self.ub, 20):
            pdfi = self.evalPDF(ui)
            if pdfi > self._max_pdf_val:
                self._max_pdf_val = pdfi
