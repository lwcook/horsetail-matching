import numpy as np
import pdb
from operator import xor


class ParameterInputError(Exception):
    '''Exception in inputs to the Parameter class'''



class Parameter():
    '''Class for handling uncertain parameters in optimization under uncertainty
    problems using horsetail matching'''

    def __init__(self, **kwargs):
        self.exists = True


class IntervalParameter(Parameter):
    '''Class for handling interval uncertain parameters in optimization
    under uncertainty problems using horsetail matching'''

    def __init(self, **kwargs):
        pass


class ProbabilisticParameter(Parameter):
    '''Class for handling probabilistic uncertain parameters in optimization
    under uncertainty problems using horsetail matching.
    
    :param string distribution: distribution type of the uncertain parameter.
    :param double mean: mean value of the distribution [default 0].
    :param double standard_deviation: standard deviation of the distribution
        [default 1].
    '''

    def __init__(self, **kwargs):
        self._process_inputs(**kwargs)

    def _process_inputs(self, **kwargs):

        dist = kwargs.pop('distribution', 'uniform')
        if not isinstance(dist, str):
            raise ParameterInputError(
                    'distribution argument should be entered as a string')

        self.distribution = dist.lower()
        featured_dists = ['uniform', 'gaussian']
        if not self.distribution.lower() in featured_dists:
            raise ParameterInputError(
                    '''Unsupported distribution type; the following
                    distributions are currently supported: ''' + \
                    ', '.join([str(e) for e in featured_dists]))

        kwitems = kwargs.items()
        if len(kwitems) == 0: ## Use defaults
            mu, std, lb, ub = 0., 1./np.sqrt(3.), -1., 1.
            return

#        if 'mean' in kwitems xor 'standard_deviation' in kwitems:
#            return ParameterInputError(
#                'Must provide mean and standard deviation together')
#        if 'lower_bound' in kwitems xor 'upper_bound' in kwitems:
#            return ParameterInputError(
#                'Must provide lower and upper bounds together')

        if self.distribution == 'uniform':
            if 'mean' in kwitems and 'standard_deviation' in kwitems:
                mu = kwargs.pop('mean', 0)
                std = kwargs.pop('standard_deviation', 1)
                lb = mu - std*np.sqrt(3.)
                ub = mu + std*np.sqrt(3.)

            elif 'lower_bound' in kwitems and 'upper_bound' in kwitems:
                lb = kwargs.pop('lower_bound')
                ub = lwargs.pop('upper_bound')
                mu = 0.5*(lb + ub)
                std = np.sqrt((ub - lb)**2/12.)

        elif self.distribution == 'gaussian':
            mu = kwargs.pop('mean', 0)
            std = kwargs.pop('standard_deviation', 1)
            if 'lower_bound' in kwitems or 'upper_bound' in kwitems:
                pass


    def evalPDF(self, u_values):
        '''Returns the PDF of the uncertain parameter evaluated at the values
        provided in u_values.

        :param iterable u_values: values of the uncertain parameter at which to
            evaluate the PDF'''
        return 0

        

