import pdb
import time
import math
import copy

import numpy as np


class HorsetailMatching(object):
    '''Class for using horsetail matching within an optimization. The main
    functionality is to evaluate the horsetail matching
    metric (and optionally its gradient) that can be used with external
    optimizers. 

    The code is written such that all arguments that can be used at the
    initialization of a HorsetailMatching object can also be set as
    attributes after creation to achieve exactly the same effect.

    :param function fqoi: function that returns the quantity of interest, it
        must take two ordered arguments - the value of the design variable
        vector and the value of the uncertainty vector.

    :param list uncertain_parameters: list of UncertainParameter objects
        that describe the uncertain inputs for the problem (they must have
        the getSample() method).

    :param function ftarget: function that returns the value of the target
        inverse CDF given a value in [0,1]. Can be a tuple that gives two
        target fuctions, one for the upper bound and one for the lower bound on
        the CDF under mixed uncertainties [default t(h) = 0]

    :param bool/function jac: Only for method = 'kernel'. Argument that
        specifies how to evaluate the gradient of the quantity of interest.
        If False no gradients are propagated, if True the fqoi should return
        a second argument g such that g_i = dq/dx_i. If a function, it should
        have the same signature as fqoi but return g. [default False]

    :param str method: method with which to evaluate the horsetil matching
        metric, can be 'empirical' or 'kernel' [default 'empirical' if
        jac is False else default 'kernel'].

    :param int samples_prob: number of samples to take from the
        probabilsitic uncertainties. [default 500]

    :param int samples_int: number of samples to take from the
        interval uncertainties. Note that under mixed uncertainties, a nested
        loop is used to evaluate the metric so the total number of
        samples will be samples_prob*samples_int (at each interval uncertainty
        sample samples_prob samples are taken from the probabilistic
        uncertainties). [default 20]

    :param list integration_points: Only for method='kernel'.
        The integration point values to use when evaluating the metric using
        kernels [by default 100 points spread over 3 times the range of
        the samples of q obtained the first time the metric is evaluated]

    :param number kernel_bandwidth: Only for method='kernel'. The bandwidth
        used in the kernel function [by default it is found the first time
        the metric is evaluated using Scott's rule]

    :param str kernel_type: Only for method='kernel'. The type of kernel to
        use, can be 'gaussian', 'uniform', or 'triangle' [default 'gaussian'].

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

    *Example Declarations*::

        >>> from horsetailmatching import HorsetailMatching,
                UncertainParameter, PolySurrogate

        >>> def myFunc(x, u): return x[0]*x[1] + u
        >>> def myGrad(x, u): return [x[1], x[0]]
        >>> def myTarg1(h): return 1-h**3
        >>> def myTarg2(h): return 2-h**3

        >>> u1 = UniformParameter()
        >>> u2 = IntervalParameter()
        >>> U = [u1, u2]
        >>> poly = PolySurrogate(dimensions=2)
        >>> poly_points = poly.getQuadraturePoints()

        >>> theHM = HorsetailMatching(myFunc, U)
        >>> theHM = HorsetailMatching(myFunc, U, jac=myGrad, method='kernel')
        >>> theHM = HorsetailMatching(myFunc, U, ftarget=myTarg1)
        >>> theHM = HorsetailMatching(myFunc, U, ftarget=(myTarg1, myTarg2))
        >>> theHM = HorsetailMatching(myFunc, U, samples_prob=500,
                samples_int = 50)
        >>> theHM = HorsetailMatching(myFunc, U, method='kernel',
                integration_points=numpy.linspace(0, 10, 100),
                kernel_bandwidth=0.01)
        >>> theHM = HorsetailMatching(myFunc, U,
                surrogate=poly.surrogate, surrogate_jac=False,
                surrogate_points=poly_points)
        >>> theHM = HorsetailMatching(myFunc, U, verbose=True,
                reuse_samples=True)

    '''

    def __init__(self, fqoi, uncertain_parameters, ftarget=None,
            jac=False, method=None,
            samples_prob=500, samples_int=20, integration_points=None,
            kernel_bandwidth=None, kernel_type='gaussian', alpha=500,
            surrogate=None, surrogate_points=None, surrogate_jac=False,
            reuse_samples=True, verbose=False):

        self.fqoi = fqoi
        self.uncertain_parameters = uncertain_parameters
        self.ftarget = ftarget
        self.jac = jac
        self.method = method # Must be done after setting jac
        self.samples_prob = samples_prob
        self.samples_int = samples_int
        self.integration_points = integration_points
        self.kernel_bandwidth = kernel_bandwidth
        self.kernel_type = kernel_type
        self.alpha = alpha
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

        self._u_int, self._u_prob = [], []
        for ii, u in enumerate(self._u_params):
            if u.is_interval_uncertainty:
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
            if (not isinstance(samples, np.ndarray) or
                    samples.shape != self._processDimensions()):
                raise TypeError('u_samples should be a np.array of size'
                        '(samples_prob, samples_int, num_uncertanities)')
        self._u_samples = samples

    @property
    def kernel_type(self):
        return self._kernel_type

    @kernel_type.setter
    def kernel_type(self, value):
        allowed_types = ['gaussian', 'uniform', 'triangle']
        if value not in allowed_types:
            raise ValueError('Kernel type must be one of'+
                    ', '.join([str(t) for t in allowed_types]))
        else:
            self._kernel_type = value


##############################################################################
##  Public Methods
##############################################################################

    def evalMetric(self, x, method=None):
        '''Evaluates the horsetail matching metric at given values of the
        design variables.

        :param iterable x: values of the design variables, this is passed as
            the first argument to the function fqoi
        :param str method: method to use to evaluate the metric ('empirical' or
            'kernel')

        :return: metric_value - value of the metric evaluated at the design
            point given by x

        :rtype: float

        *Example Usage*::

            >>> def myFunc(x, u): return x[0]*x[1] + u
            >>> u1 = UniformParameter()
            >>> theHM = HorsetailMatching(myFunc, u)
            >>> x0 = [1, 2]
            >>> theHM.evalMetric(x0)

        '''

        if self.verbose:
            print('----------')
            print('At design: ' + str(x))

        # Make sure dimensions are correct
        u_sample_dimensions = self._processDimensions()

        self._N_dv = len(_makeIter(x))

        if method is None:
            method = self.method

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
        if method.lower() == 'empirical':
            if self.jac:
                raise TypeError( 'Empicial method does not support gradients')
            else:
                return self._evalMetricEmpirical(q_samples)
        elif method.lower() == 'kernel':
            return self._evalMetricKernel(q_samples, grad_samples)
        else:
            raise ValueError('Unsupported metric evalation method')

    def getHorsetail(self):
        '''Function that gets vectors of the horsetail plot at the last design
        evaluated.

        :return: upper_curve, lower_curve, CDFs - returns three parameters,
            the first two are tuples containing pairs of x/y vectors of the
            upper and lower bounds on the CDFs (the horsetail plot). The
            third parameter is a list of x/y tuples for individual CDFs
            propagated at each sampled value of the interval uncertainties

        *Example Usage*::

            >>> def myFunc(x, u): return x[0]*x[1] + u
            >>> u = UniformParameter()
            >>> theHM = HorsetailMatching(myFunc, u)
            >>> (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()
            >>> matplotlib.pyplot(x1, y1, 'b')
            >>> matplotlib.pyplot(x2, y2, 'b')
            >>> for (x, y) in CDFs:
            ...     matplotlib.pyplot(x, y, 'k:')
            >>> matplotlib.pyplot.show()

        '''

        if hasattr(self, '_ql'):

            ql, qu, hl, hu = self._ql, self._qu, self._hl, self._hu
            qh, hh = self._qh, self._hh

            if self.integration_points is not None:
                ql, hl = _appendPlotArrays(ql, hl, self.integration_points)
                qu, hu = _appendPlotArrays(qu, hu, self.integration_points)

            CDFs = []
            for qi, hi in zip(qh, hh):
                CDFs.append((qi, hi))

            upper_curve = (qu, hu)
            lower_curve = (ql, hl)
            return upper_curve, lower_curve, CDFs

        else:
            raise ValueError('''The metric has not been evaluated at any
                    design point so the horsetail does not exist''')

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
            D_u += (1./self.samples_prob)*(qui - self._ftarg_u(hui))**2
            D_l += (1./self.samples_prob)*(qli - self._ftarg_l(hli))**2

        dhat = np.sqrt(D_u + D_l)
        self._ql, self._qu, self._hl, self._hu = q_l, q_u, h_l, h_u
        self._qh, self._hh = q_htail, h_htail
        return dhat

    def _evalMetricKernel(self, q_samples, grad_samples=None):

        # If kernel bandwidth not specified, find it using Scott's rule
        if self.kernel_bandwidth is None:
            if abs(np.max(q_samples) - np.min(q_samples)) < 1e-6:
                bw = 1e-6
            else:
                bw = 0.1*((4/(3.*q_samples.shape[1]))**(1/5.)
                          *np.std(q_samples[0,:]))
            self.kernel_bandwidth = bw
        else:
            bw = self.kernel_bandwidth

        ## Initalize arrays and prepare calculation
        if self.integration_points is None:
            q_min = np.amin(q_samples)
            q_max = np.amax(q_samples)
            q_range = q_max - q_min
            qis = np.linspace(q_min - q_range, q_max + q_range, 100)
            self.integration_points = qis
        else:
            qis = self.integration_points
        N_quad = len(qis)
        M_prob = self.samples_prob
        M_int = self.samples_int

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
                Kcdf, Kprime = _kernel(rmat, M_prob, bw=bw,
                        ktype=self.kernel_type, bGrad=True)
                for ix in np.arange(self._N_dv):
                    grad_js = grad_samples[ii, :, ix]
                    fht_grad[ii, :, ix] = Kprime.dot(-1*grad_js)
            else:
                Kcdf = _kernel(rmat, M_prob, bw=bw, ktype=self.kernel_type,
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
            tu_pr = np.array([_finDiff(self._ftarg_u, hi) for hi in hu])
            tl_pr = np.array([_finDiff(self._ftarg_l, hi) for hi in hl])
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
        if self.surrogate_points is None:
            N_u = len(self._u_prob) + len(self._u_int)
            mesh = np.meshgrid(*[np.linspace(-1, 1, 5) for n in np.arange(N_u)],
                    copy=False)
            u_sparse = np.vstack([m.flatten() for m in mesh]).T
        else:
            u_sparse = self.surrogate_points

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
            for iu, u in enumerate(u_sparse):
                if isinstance(self.jac, bool) and self.jac:
                    q_sparse[iu], g_sparse[iu, :] = self.fqoi(x, u)
                else:
                    q_sparse[iu] = self.fqoi(x, u)
                    g_sparse[iu, :] = self.jac(x, u)

            if not self.surrogate_jac:
                fpartial = [lambda u: 0 for _ in np.arange(self._N_dv)]
                surr_qoi = self.surrogate(u_sparse, q_sparse)
                for k in np.arange(self._N_dv):
                    fpartial[k] = self.surrogate(u_sparse, g_sparse[:, k])
                def surr_grad(u):
                    return [f(u) for f in fpartial]
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

            # Sample over interval uncertainties,
            # Then at each value sample over the probabilistic uncertainties
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
            if self.verbose:
                print('Re-using stored samples')
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
        q_samples = np.zeros([self.samples_int, self.samples_prob])
        if not jac:
            for ii in np.arange(q_samples.shape[0]):
                for jj in np.arange(q_samples.shape[1]):
                    q_samples[ii, jj] = fqoi(u_samples[ii, jj])
        else:
            grad_samples = np.zeros([self.samples_int, self.samples_prob,
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
            u_sample_dim = (self.samples_int, self.samples_prob, N_u)

        # Probabilistic uncertainties
        elif len(self._u_int) == 0:
            self.samples_int = 1
            u_sample_dim = (1, self.samples_prob, N_u)

        # Interval Uncertainties
        elif len(self._u_prob) == 0:
            self.samples_prob = 1
            u_sample_dim = (self.samples_int, 1, N_u)
            self.kernel_bandwidth = 1e-3

        return u_sample_dim

##############################################################################
##  Private functions
##############################################################################

def _extalg(xarr, alpha=100):
    '''Given an array xarr of values, smoothly return the max/min'''
    return sum(xarr * np.exp(alpha*xarr))/sum(np.exp(alpha*xarr))

def _extgrad(xarr, alpha=100):
    '''Given an array xarr of values, return the gradient of the smooth min/max
    swith respect to each entry in the array'''
    term1 = np.exp(alpha*xarr)/sum(np.exp(alpha*xarr))
    term2 = 1 + alpha*(xarr - _extalg(xarr, alpha))

    return term1*term2

def _ramp(x, width):
    return _minsmooth(1, _maxsmooth(0, (x - width/2)*(1/width)))

def _trint(x, width):
    w = width/2.
    xb = _maxsmooth(-w, _minsmooth(x, w))
    y1 = 0.5 + xb/w + xb**2/(2*w**2)
    y2 = xb/w - xb**2/(2*w**2)
    return _minsmooth(y1, 0.5) + _maxsmooth(y2, 0.0)

def _minsmooth(a, b, eps=0.0000):
    return 0.5*(a + b - np.sqrt((a-b)**2 + eps**2))

def _maxsmooth(a, b, eps=0.0000):
    return 0.5*(a + b + np.sqrt((a-b)**2 + eps**2))

def _step(x):
    return 1 * (x > 0)

def _kernel(points, M=None, bw=None, ktype='gauss', bGrad=False):

    # NB make evaluations matrix compatible
    if ktype == 'gauss' or ktype == 'gaussian':
        KernelMat = np.zeros(points.shape)
        for ir in np.arange(points.shape[0]):
            for ic in np.arange(points.shape[1]):
                KernelMat[ir, ic] = (1./M)*((1. +
                    math.erf((points[ir, ic]/bw)/math.sqrt(2.)))/2.)

    elif ktype == 'uniform' or ktype == 'uni':
        KernelMat = (1./M)*_ramp(points, width=bw*np.sqrt(12))
    elif ktype == 'triangle' or ktype == 'tri':
        KernelMat = (1./M)*_trint(points, width=bw*2.*np.sqrt(6))

    if bGrad:
        if ktype == 'gauss' or ktype == 'gaussian':
            const_term = 1.0/(M * np.sqrt(2*np.pi*bw**2))
            KernelGradMat = const_term * np.exp(-(1./2.) * (points/bw)**2)
        elif ktype == 'uniform' or ktype == 'uni':
            width = bw*np.sqrt(12)
            const = (1./M)*(1./width)
            KernelGradMat = const*(_step(points+width/2) -
                                   _step(points-width/2))
        elif ktype == 'triangle' or ktype == 'tri':
            width = bw*2.*np.sqrt(6)
            const = (1./M)*(2./width)
            KernelGradMat = const*(_ramp(points+width/4, width/2) -
                                   _ramp(points-width/4, width/2))

        return KernelMat, KernelGradMat
    else:
        return KernelMat

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

def _appendPlotArrays(q, h, integration_points):
    q = np.insert(q, 0, q[0])
    h = np.insert(h, 0, 0)
    q = np.insert(q, 0, min(integration_points))
    h = np.insert(h, 0, 0)
    q = np.append(q, q[-1])
    h = np.append(h, 1)
    q = np.append(q, max(integration_points))
    h = np.append(h, 1)
    return q, h

def _finDiff(fobj, dv, f0=None, eps=10**-6):

    if f0 is None:
        f0 = fobj(dv)

    fbase = copy.copy(f0)
    fnew = fobj(dv + eps)
    return float((fnew - fbase)/eps)

def _makeIter(x):
    try:
        iter(x)
        return [xi for xi in x]
    except:
        return [x]
