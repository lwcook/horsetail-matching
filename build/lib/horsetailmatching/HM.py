import numpy as np
import copy
import pdb
import random as rdm
import time
import scipy.special as scp
import scipy.stats as scs
import matplotlib.pyplot as plt
from scipy import optimize as scipyopt

import utilities as utils
import targets as targ


class HM():

    def __init__(self, fobj, **kwargs):
        '''
        - fobj: quantity of interest.
        - fgrad: function returning the gradient of the quantity of interest
            or a boolean stating whether fobj also returns gradient;
            if False it is found with finite differencing.
        - t1, t2: target inverse CDFs.
        - ualdim, uepdim: number of aleatory and epistemic uncertainties.
        - n_sample, n_quad: no. sample and quadrature points in HM'''

        self.fobj = fobj  # fobj should take two inputs: dv, u.
        self.fgrad = kwargs.setdefault('fgrad', False)
#       self.udict = kwargs.setdefault('udict', {1: 'uniform', 2: 'interval'})
        self.t1 = kwargs.setdefault(
            't1', lambda x: targ.target_u(x, 0, std=0.001, bInverse=True))
        self.t2 = kwargs.setdefault(
            't2', lambda x: targ.target_u(x, 0, std=0.001, bInverse=True))
        self.ualdim = kwargs.setdefault('ualdim', 1)  # First ualdim aleatory
        self.uepdim = kwargs.setdefault('uepdim', 1)  # Last uepdim epistemic

        # HM integration variables
        self.n_sample = int(kwargs.setdefault('n_sample', 10**4))
        self.n_quad = int(kwargs.setdefault('n_quad', 10**3))
        self.n_ep = int(kwargs.setdefault('n_ep', 20))
        self.poly_order = int(kwargs.setdefault('poly_order', 3))
        self.p = int(kwargs.setdefault('p', 2))
        self.q_lo = kwargs.setdefault('q_lo', 0)
        self.q_hi = kwargs.setdefault('q_hi', 10)
        self.ujs = None  # This is overwritten when an optimziation begins
        self.mean = None
        self.var = None

        # Type of kernel
        self.ktype = kwargs.setdefault('ktype', 'gaussian')
        self.bw = kwargs.setdefault('bw', 0.05)

        # Mixed propagation parameters
        self.alpha = kwargs.setdefault('alpha', 100)

        self.lf_evals = 0
        self.hf_evals = 0
        self.tot_evals = 0

        self.bPlot = kwargs.setdefault('bPlot', False)
        self.bTails = True
        self.bTarg = True
        self.fig = None
        self.drawStyle = '-'

    def hm_metric(self, dv, **kwargs):
        """ Returns the horsetail matching d_p metric for a given design
            Uses the targets specified in self.t1 and selfT2inv
            - dv: the design variables """

        if self.uepdim + self.ualdim == 1:
            u_sparse = np.linspace(-1, 1, 6)
        elif self.uepdim + self.ualdim == 2:
            x, y = np.linspace(-1, 1, 4), np.linspace(-1, 1, 4)
            X, Y = np.meshgrid(x, y, copy=False)
            u_sparse = np.array(zip(X.flatten(), Y.flatten()))
        elif self.uepdim + self.ualdim == 3:
            x, y = np.linspace(-1, 1, 4), np.linspace(-1, 1, 4)
            z = np.linspace(-1, 1, 4)
            X, Y, Z = np.meshgrid(x, y, z, copy=False)
            u_sparse = np.array(zip(X.flatten(), Y.flatten(), Z.flatten()))
        elif self.uepdim + self.ualdim == 4:
            x, y = np.linspace(-1, 1, 4), np.linspace(-1, 1, 4)
            z, w = np.linspace(-1, 1, 4), np.linspace(-1, 1, 4)
            X, Y, Z, W = np.meshgrid(x, y, z, w, copy=False)
            u_sparse = np.array(zip(X.flatten(), Y.flatten(),
                                    Z.flatten(), W.flatten()))

        q_sparse, grad_sparse = utils.eval_quad_points(
            u_sparse, dv, self.fobj, nu=self.ualdim+self.uepdim,
            f_grad=self.fgrad, bGrad=True, eps=10**-6)

        N = kwargs.setdefault('N', self.n_quad)
        vqx = np.linspace(0, 1, N, endpoint=True)
        geom = lambda x: 0.6666*x**3 + 0.3333*x
        vqy = geom(vqx)
        vq_i = (self.q_lo+vqy*(self.q_hi-self.q_lo)).reshape([N, 1])

        if self.uepdim == 0:
            return self.metric_prob(u_sparse, q_sparse, grad_sparse,
                                    qis=vq_i, ndv=len(dv), **kwargs)

        elif self.ualdim == 0:  # Interval Uncertainties
            return self.metric_int(u_sparse, q_sparse, grad_sparse,
                                   qis=vq_i, ndv=len(dv), **kwargs)

        else:  # Mixed uncertainty, must be >= 2 uncertainties
            return self.metric_mix(u_sparse, q_sparse, grad_sparse,
                                   qis=vq_i, ndv=len(dv), **kwargs)

    def metric_prob(self, u_sparse, q_sparse, grad_sparse, **kwargs):
        M = kwargs.setdefault('M', self.n_sample)
        N = kwargs.setdefault('N', self.n_quad)
        ndv = kwargs['ndv']
        bGrad = kwargs.setdefault('bGrad', False)
        qis0 = np.linspace(self.q_lo, self.q_hi, N).reshape([N, 1])
        qis = kwargs.setdefault('qis', qis0)

        # Use SAA - do not resample at evey optimization iteration
        if self.ujs is not None:
            ujs = self.ujs
        else:
            prob = Probability(dims=self.ualdim)
            ujs = prob.find_sample_points(M, 'uniform') \
                .reshape([M, self.ualdim])
            self.ujs = ujs

        qjs = utils.surrogate(u_sparse, q_sparse, ujs,
                             dims=self.ualdim).reshape([1, M])

        self.mean = np.mean(qjs)
        self.var = np.var(qjs)

        Kcdf, Kprime = utils.kernel(qis-qjs, M, bw=self.bw, bGrad=True)

        his = Kcdf.dot(np.ones([M, 1])).reshape([N, 1])

        tis = np.array([float(self.t1(hi)) for hi in his]).reshape([N, 1])

        D1 = float(self.matrix_integration(qis, his, tis))

        dp = (D1 + D1)**(1.0/self.p)

        if bGrad:
            D1g = np.zeros(ndv)
            for kdv in range(ndv):  # d/dxk - gradient wrt 1 dv
                q_dx = utils.surrogate(u_sparse, grad_sparse[:, kdv],
                                       ujs, dims=self.ualdim).reshape([M, 1])

                h_dx = Kprime.dot(-1*q_dx)

                D1g[kdv] = self.matrix_gradient(qis, his, tis, h_dx)

            grad = (1./self.p)*((D1+D1)**(1./self.p - 1.))*(D1g+D1g)

        if self.bPlot:
#            utils.mpl2tex()
#            plt.figure(figsize=(8, 6))
            plt.ion()
            plt.plot(qis, his, 'k')
#            plt.plot(tis, his, c='b')
            plt.xlabel('q')
            plt.ylabel('h')
#            plt.xlim([200, 400])
            plt.ylim([0, 1])
#            plt.tight_layout()
#            utils.savefig('horsetail_demo_prob')
#            plt.show()
            plt.draw()

        if not bGrad:
            return dp
        else:
            return dp, [g for g in grad]

    def metric_mix(self, u_sparse, q_sparse, grad_sparse, **kwargs):
        M = kwargs.setdefault('M', self.n_sample)
        N = kwargs.setdefault('N', self.n_quad)
        ndv = kwargs['ndv']
        bGrad = kwargs.setdefault('bGrad', False)
        qis0 = np.linspace(self.q_lo, self.q_hi, N).reshape([N, 1])
        qis = kwargs.setdefault('qis', qis0)

        alpha0 = self.alpha
        alpha = kwargs.setdefault('alpha', alpha0)

        # Outer loop: sample over epistemic uncertainties
        n_ep = self.n_ep
        u_dim = self.ualdim + self.uepdim

        u_eps = utils.gridsample(n_ep, self.uepdim)
        n_ep = len(u_eps)
        h_eps = np.zeros([n_ep, N])
        h_eps_dx = np.zeros([n_ep, N, ndv])

        # Use SAA - reuse same probabilistic samples
        if self.ujs is not None:
            ujs_al = self.ujs
        else:
            prob = Probability(dims=self.ualdim)
            ujs_al = prob.find_sample_points(M, 'uniform') \
                .reshape([M, self.ualdim])
            self.ujs = ujs_al

        # Inner loop: for each u_ep, propagate a CDF
        for iep, u_ep in enumerate(u_eps):
            ujs = [[_ for _ in uj[0:self.ualdim]] + u_ep for uj in ujs_al]
            ujs = np.array(ujs)

            qjs = utils.surrogate(u_sparse, q_sparse, ujs,
                                 dims=u_dim).reshape([1, M])

            Kcdf, Kprime = utils.kernel(qis-qjs, M, bw=self.bw, bGrad=True)

            his = Kcdf.dot(np.ones([M, 1])).reshape([N, 1])
            h_eps[iep, :] = his.reshape([N])

            if bGrad:

                for kdv in range(ndv):  # d/dxk - gradient wrt 1 dv
                    q_dx = utils.surrogate(u_sparse, grad_sparse[:, kdv],
                                           ujs, dims=u_dim).reshape([M, 1])

                    h_dx = Kprime.dot(-1*q_dx)

                    h_eps_dx[iep, :, kdv] = h_dx.flatten()

        hmax = np.array([utils.extalg(h_eps[:, i], alpha) for i in range(N)])
        hmin = np.array([utils.extalg(h_eps[:, i], -alpha) for i in range(N)])

        t1 = np.array([self.t1(hi) for hi in hmax]).reshape([N, 1])
        t2 = np.array([self.t2(hi) for hi in hmin]).reshape([N, 1])

        D1 = float(self.matrix_integration(qis, hmax, t1))
        D2 = float(self.matrix_integration(qis, hmin, t2))

        dp = (D1 + D2)**0.5

        if bGrad:

            hmaxprime = np.zeros([n_ep, N])
            hminprime = np.zeros([n_ep, N])
            for i in range(N):
                hmaxprime[:, i] = utils.extgrad(h_eps[:, i], alpha)
                hminprime[:, i] = utils.extgrad(h_eps[:, i], -alpha)

            D1dx, D2dx = np.zeros([ndv]), np.zeros([ndv])
            for kdv in range(ndv):
                hmax_dx, hmin_dx = np.zeros([N]), np.zeros([N])
                for i in range(N):

                    hmax_dx[i] = np.dot(hmaxprime[:, i], h_eps_dx[:, i, kdv])
                    hmin_dx[i] = np.dot(hminprime[:, i], h_eps_dx[:, i, kdv])

                D1dx[kdv] = self.matrix_gradient(qis, hmax, t1, hmax_dx)
                D2dx[kdv] = self.matrix_gradient(qis, hmin, t2, hmin_dx)

            grad = 0.5*((D1+D2)**(-0.5))*(D1dx+D2dx)

        if self.bPlot:
            if self.fig is None:
                fig1 = plt.figure()
                plt.ion()
                self.fig = fig1
            plt.figure(self.fig.number)
#            plt.cla()
#            plt.ion()
#            utils.mpl2tex()
#            plt.figure(self.fig.number)
            col = [0.8, 0.8, 0.8]
            # plt.figure(figsize=(8, 6))
            if self.bTails:
                for iep in range(n_ep):
                    plt.plot(qis, h_eps[iep, :], c=col)
            lstyle = 'k' + self.drawStyle
            if self.drawStyle == '--':
                plt.plot(qis, hmax, lstyle, dashes=(4, 2))
                plt.plot(qis, hmin, lstyle, dashes=(4, 2))
            else:
                plt.plot(qis, hmax, lstyle)
                plt.plot(qis, hmin, lstyle)
            if self.bTarg:
                plt.plot(t1, hmax, 'k--', dashes=(4, 2))
                plt.plot(t2, hmin, 'k--', dashes=(4, 2))
            plt.xlabel('q')
            plt.ylabel('h')
#            plt.xlim([200, 800])
            plt.ylim([0, 1])
#            plt.tight_layout()
#            utils.savefig('horsetail_demo_alpha' + str(alpha))
#            plt.show(block=False)
#            plt.draw()
#            plt.pause(0.05)

        if not bGrad:
            return dp
        else:
            return dp, [g for g in grad]

    def metric_int(self, u_sparse, q_sparse, grad_sparse, **kwargs):
        M = kwargs.setdefault('M', self.n_sample)
        ndv = kwargs.setdefault('ndv')
        print(ndv)

        prob = Probability(dims=2)
        u_eps = prob.find_sample_points(M, 'uniform').reshape([M, 2])
        q_eps = utils.surrogate(u_sparse, q_sparse, u_eps,
                             dims=2).reshape([1, M])

        qmax = utils.extalg(q_eps[0, :], 10)
        qmin = utils.extalg(q_eps[0, :], -10)

        D1 = (qmin - self.t1(0.5))**self.p
        D2 = (qmax - self.t2(0.5))**self.p
        dp = (D1 + D2)**(1.0/self.p)

        return dp

    def matrix_integration(self, q, h, t, **kwargs):
        ''' Returns the dp metric for a single horsetail
        curve at a given value of the epistemic uncertainties'''

        N = kwargs.setdefault('N', self.n_quad)

        # correction if CDF has gone out of trapezium range
        if h[-1] < 0.9: h[-1] = 1.0

        W = np.zeros([N, N])
        for i in range(N):
            W[i, i] = 0.5*(h[min(i+1, N-1)] - h[max(i-1, 0)])

        dp = (q - t).T.dot(W).dot(q - t)

        return dp

    def matrix_gradient(self, q, h, t, h_dx, **kwargs):
        ''' Returns the gradient with respect to a single variable'''

        N = kwargs.setdefault('N', self.n_quad)

        W = np.zeros([N, N])
        Wprime = np.zeros([N, N])
        for i in range(N):
            W[i, i] = 0.5*(h[min(i+1, N-1)] - h[max(i-1, 0)])
            Wprime[i, i] = \
                0.5*(h_dx[min(i+1, N-1)] - h_dx[max(i-1, 0)])

        tprime = np.zeros([N, 1])
        for i in range(N):
            Tgrad = utils.finite_diff(
                lambda h_: self.t1(h_), [h[i]], eps=10**-6)
            tprime[i] = Tgrad*h_dx[i]

        grad = 2.0*(q - t).T.dot(W).dot(-1.0*tprime)\
            + (q - t).T.dot(Wprime).dot(q - t)

        return float(grad)


class Probability():

    def __init__(self, dims=1):
        self.dims = dims

    def sample(self, sample_num = None, random_type = 'uniform',alpha=2,beta=2): 
        if sample_num == None:  sample_num = self.sample_num
        self.sample_num = int(sample_num)

        if isinstance(random_type,list):
            sample_points = self.compound_sample_points(sample_num,random_type)
        else:
            sample_points = self.find_sample_points(sample_num,random_type,alpha=alpha,beta=beta)

        results = np.zeros([self.sample_num,1],float)

        for ii in range(len(self.sample_points)):
            results[ii] = self.eval_response(sample_points[ii,:])

        self.sample_data = results 
        return self.sample_data

    def compound_sample_points(self, sample_num = None, random_type = 'uniform'):
        if isinstance(random_type, list) and len(random_type) == self.dims:
            sample_points = np.zeros([sample_num,self.dims])
            _dimtemp = self.dims
            for ii in range(len(random_type)):
                self.dims = 1
                _ = self.find_sample_points(sample_num=sample_num, random_type= random_type[ii])
                sample_points[:,ii] = _.reshape(sample_num)

            self.dims = _dimtemp

        return sample_points

    def find_sample_points(self, sample_num = None, random_type = 'uniform',alpha=2,beta=2):
        if sample_num == None:  sample_num = self.sample_num
        self.sample_num = int(sample_num)

        if random_type == 'gaussian':
            sample_points = self._gauss_sample(sample_num)
        if random_type == 'beta':
            sample_points = self._beta_sample(sample_num,alpha=alpha,beta=beta)
        elif random_type == 'uniform' or random_type == 'LHS': # uniform distribution using latin hypercube sampling
            sample_points = self._LHS(-1.*np.ones(self.dims), 1*np.ones(self.dims), self.sample_num)

        #pdb.set_trace()

        self.sample_points = sample_points
        return sample_points


    def _LHS(self, lbvec, ubvec, sample_num): # Latin Hypercube Sampling

        # Only returns the sample points 
        lbvec = np.array(lbvec).reshape(np.array(lbvec).size)
        ubvec = np.array(ubvec).reshape(np.array(ubvec).size)
        if lbvec.shape != ubvec.shape:
            print('Error, shapes are not consistent')

        sample_points = np.zeros([sample_num,self.dims])
        permutations = np.zeros([sample_num,self.dims],int)

        # Using a uniform distribution
        srange = np.zeros(lbvec.shape[0])
        for idim in range(lbvec.shape[0]):

            srange[idim] = ubvec[idim] - lbvec[idim]
            segment_size = srange[idim] / float(sample_num)
            for isample in range(0,sample_num):

                segment_min = lbvec[idim] + isample*segment_size
                sample_points[isample,idim] = segment_min + np.random.uniform(0,segment_size)

            permutations[:,idim] = np.random.permutation(sample_num)

        sample_points_temp = sample_points*0
        for isample in range(0,sample_num):
            for idim in range(0,self.dims):
                sample_points_temp[isample,idim] = sample_points[permutations[isample,idim],idim]
        sample_points = sample_points_temp

        return sample_points

    def _gauss_sample(self, sample_num = 'None'):
        #pdb.set_trace()
        if sample_num == 'None': sample_num = 100
        sample_points = np.zeros([sample_num,self.dims])

        for idim in range(self.dims):
            for isample in range(0,sample_num):
                sample_points[isample,idim] = np.random.normal(scale=math.sqrt(0.5))

        return sample_points

    def _beta_sample(self, sample_num = 'None',alpha=1,beta=1):
        if sample_num == 'None': sample_num = 100
        sample_points = np.zeros([sample_num,self.dims])

        l, r = -1,1
        for idim in range(self.dims):
            for isample in range(0,sample_num):
                sample_points[isample,idim] = l + (r-l)*np.random.beta(alpha,beta)
        #pdb.set_trace()

        return sample_points
