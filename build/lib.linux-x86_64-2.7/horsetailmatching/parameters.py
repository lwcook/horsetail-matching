import numpy as np
import pdb
from operator import xor
import utilities as utils
from inspect import isfunction
import warnings


class Parameter():
    '''Class for handling uncertain parameters in optimization under uncertainty
    problems using horsetail matching'''

    def __init__(self, **kwargs):
        self.exists = True


class IntervalParameter(Parameter):
    '''Class for handling interval uncertain parameters in optimization
    under uncertainty problems using horsetail matching.

    :param double lower_bound: lower_bound of the interval [default -1]
    :param double upper_bound: upper_bound of the interval [default 1]
    '''

    def __init__(self, lower_bound=-1*float("inf"), upper_bound=float("inf")):
        self.lb = lower_bound
        self.ub = upper_bound



class ProbabilisticParameter(Parameter):
    '''Class for handling probabilistic uncertain parameters in optimization
    under uncertainty problems using horsetail matching.

    :param str distribution: distribution type of the uncertain parameter.
        Supported distributions are uniform, gaussian, custom (must provide a
        function to the pdf argument) [default uniform]
    :param double mean: mean value of the distribution [default 0]
    :param double standard_deviation: standard deviation of the distribution
        [default 1]
    :param function pdf: pdf function to use with custom distribution
    :param double lower_bound: lower bound of the distribution (overridden by
        mean and standard_devaition inputs) [default -1]
    :param double upper_bound: upper bound of the distribution (overridden by
        mean and standard_devaition inputs) [default 1]
    '''

    featured_dists = ['uniform', 'gaussian', 'custom']

    def __init__(self, distribution='uniform', mean=0, standard_deviation=1,
            pdf=None, lower_bound=None, upper_bound=None):
        self.distribution = distribution.lower()
        self.mu, self.std = mean, standard_deviation
        self.lb, self.ub = lower_bound, upper_bound
        self.fpdf = pdf

        if self.distribution == 'uniform' and self.lb is not None:
            self.mu = 0.5*(self.ub + self.lb)
            self.std = np.sqrt((self.ub - self.lb)**2/12.)

        self._check_attributes()

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
        if self.distribution == 'uniform':
            lb = self.mu - np.sqrt(3)*self.std
            ub = self.mu + np.sqrt(3)*self.std
            return 1./(ub - lb)

        elif self.distribution == 'gaussian':
            const = 1./(2*np.pi*self.std**2)
            return const*np.exp(-(u - self.mu)**2/(2*self.std**2))

        elif self.distribution == 'custom':
            return self.fpdf(u)

    def _check_attributes(self):

        if not self.distribution in self.featured_dists:
            raise ValueError(
                    '''Unsupported distribution type; the following
                    distributions are currently supported: ''' +
                    ', '.join([str(e) for e in self.featured_dists]))

        if xor(self.lb is None, self.ub is None):
            return ValueError( 'Must give lower and upper bounds together')

        elif self.distribution == 'gaussian' and self.ub is not None:
            warnings.warn("Gaussian is unbounded, ignoring bounds")

        elif self.distribution == 'custom':
            if self.fpdf is None:
                raise ValueError('''A pdf function should be provided with
                    custom distribution''')
            elif not isfunction(self.fpdf):
                raise ValueError('''Given pdf should be a function''')
