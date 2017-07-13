import numpy as np
import math
import pdb


class PolySurrogate(object):
    '''Class for creating surrogate models using non-intrusive polynomial
    chaos.

    :param int dimensions: number of dimensions of the polynomial expansion

    :param int order: order of the polynomial expansion [default 3]

    :param str/list poly_type: string of the type of polynomials to use in the
        expansion, or list of strings where each entry in the list is the type
        of polynomial to use in the corresponding dimension. Supported
        polynomial types are legendre and gaussian. [default legendre]

    *Example Declaration*::

        >>> thePC = PolySurrogate(dimensions=3)
        >>> thePC = PolySurrogate(dimensions=3, order=3)
        >>> thePC = PolySurrogate(dimensions=3, order=3, poly_type='legendre')

    '''

    def __init__(self, dimensions, order=3, poly_type='legendre'):

        self.dims = dimensions
        self.P = int(order) + 1

        if isinstance(poly_type, basestring):
            self.poly_types = [poly_type for _ in np.arange(self.dims)]
        else:
            self.poly_types = _makeIter(poly_type)
        self.J_list = [_define_poly_J(p, self.P) for p in self.poly_types]

        imesh = np.meshgrid(*[np.arange(self.P) for d in np.arange(self.dims)])
        self.index_polys = np.vstack([m.flatten() for m in imesh]).T 
        self.N_poly = len(self.index_polys)
        self.coeffs = np.zeros([self.P for __ in np.arange(self.dims)])

    def surrogate(self, u_sparse, q_sparse):
        '''Combines the train and predict methods to create a surrogate
        model function fitted to the input/output combinations given in
        u_sparse and q_sparse.

        :param numpy.ndarray u_sparse: input values at which the output
            values are obtained.  Must be the same as the qaudrature
            points defined by the getQuadraturePoints method.
        :param numpy.ndarray q_sparse: output values corresponding
            to the input values given in u_sparse to which the
            surrogate is fitted

        :return: surrogate model fitted to u_sparse and q_sparse

        :rtype: function

        *Sample Usage*::

            >>> thePC = PolySurrogate(dimensions=2)
            >>> U = thePC.getQuadraturePoints()
            >>> Q = [myFunc(u) for u in U]
            >>> surrogateFunc = thePC.surrogate(U, Q)

        '''
        self.train(q_sparse)
        def model(u):
            return self.predict(u)
        return model

    def predict(self, u):
        '''Predicts the output value at u from the fitted polynomial expansion.
        Therefore the method train() must be called first.

        :param numpy.ndarray u: input value at which to predict the output.

        :return: q_approx - the predicted value of the output at u

        :rtype: float

        *Sample Usage*::

            >>> thePC = PolySurrogate(dimensions=2)
            >>> U = thePC.getQuadraturePoints()
            >>> Q = [myFunc(u) for u in U]
            >>> thePC.train(U, Q)
            >>> thePC.predict([0, 1])

        '''
        y, ysub = 0, np.zeros(self.N_poly)
        for ip in range(self.N_poly):
            inds = tuple(self.index_polys[ip])
            ysub[ip] = self.coeffs[inds]*eval_poly(u, inds, self.J_list)
            y += ysub[ip]

        self.response_components = ysub
        return y

    def train(self, fpoints):
        '''Trains the polynomial expansion.

        :param numpy.ndarray fpoints: output values corresponding to the
            quadrature points given by the getQuadraturePoints method to
            which the expansion should be trained.

        *Sample Usage*::

            >>> thePC = PolySurrogate(dimensions=2)
            >>> U = thePC.getQuadraturePoints()
            >>> Q = [myFunc(u) for u in U]
            >>> thePC.train(U, Q)
            >>> predicted_q = thePC.predict([0, 1])

        '''
        self.coeffs = 0*self.coeffs

        upoints, wpoints = self.getQuadraturePointsAndWeights()

        for ipoly in np.arange(self.N_poly):

            inds = tuple(self.index_polys[ipoly])
            coeff = 0.0
            for (u, q, w) in zip(upoints, fpoints, wpoints):
                coeff += eval_poly(u, inds, self.J_list)*q*np.prod(w)

            self.coeffs[inds] = coeff
        return None

    def getQuadraturePointsAndWeights(self):
        '''Gets the quadrature points and weights for gaussian quadrature
        integration of inner products from the definition of the polynomials in
        each dimension.


        :return: (u_points, w_points) - np.ndarray of shape
            (num_polynomials, num_dimensions) and a np.ndarray of size
            (num_polynomials)

        :rtype: (np.ndarray, np.ndarray)
        '''

        qw_list, qp_list = [], []
        for ii in np.arange(len(self.J_list)):

            d, Q = np.linalg.eig(self.J_list[ii])
            qp, qpi = d[np.argsort(d)].reshape([d.size, 1]), np.argsort(d)
            qw = (Q[0, qpi]**2).reshape([d.size, 1])

            qw_list.append(qw)
            qp_list.append(qp)

        umesh = np.meshgrid(*qp_list)
        upoints = np.vstack([m.flatten() for m in umesh]).T

        wmesh = np.meshgrid(*qw_list)
        wpoints = np.vstack([m.flatten() for m in wmesh]).T

        return upoints, wpoints

    def getQuadraturePoints(self):
        '''Gets the quadrature points at which the output values must be found
        in order to train the polynomial expansion using gaussian quadrature.

        :return: upoints - a np.ndarray of size (num_polynomials, num_dimensions)

        :rtype: np.ndarray
        '''
        upoints, _ = self.getQuadraturePointsAndWeights()
        return upoints

## --------------------------------------------------------------------------
## Private funtions for polynomials
## --------------------------------------------------------------------------

def eval_poly(uvec, nvec, Jvec):
    '''Evaluate multi-dimensional polynomials through tensor multiplication.

    :param list uvec: vector value of the uncertain parameters at which to evaluate the
        polynomial

    :param list nvec: order in each dimension at which to evaluate the polynomial

    :param list Jvec: Jacobi matrix of each dimension's 1D polynomial

    :return: poly_value - value of the polynomial evaluated at uvec

    :rtype: float

    '''
    us = _makeIter(uvec)
    ns = _makeIter(nvec)
    Js = _makeIter(Jvec)
    return np.prod([_eval_poly_1D(u, n, J) for u, n, J in zip(us, ns, Js)])

def _eval_poly_1D(s, k, Jmat):
    if k == -1:
        return 0.0
    elif k == 0:
        return 1.0
    else:
        ki = k-1
        beta_k = float(Jmat[ki+1, ki])
        alpha_km1 = float(Jmat[ki, ki])

        if k == 1:
            beta_km1 = 0.
        else:
            beta_km1 = float(Jmat[ki, ki-1])

        return (1.0/float(beta_k))*(
                (s - alpha_km1)*_eval_poly_1D(s, k-1, Jmat) -
                beta_km1*_eval_poly_1D(s, k-2, Jmat))

def _define_poly_J(typestr, order, a=1, b=1):

    n = order
    # Define ab, the matrix of alpha and beta values
    # These are recurrence coefficients
    if typestr == 'legendre' or typestr == 'uniform':
        l, r = -1, 1
        o = l + (r-l)/2.0
        ab = np.zeros([n, 2],float)
        if n > 0:
            ab[0, 0], ab[0, 1] = o,1

        for k in np.arange(2, n+1, 1):
            ik, ab[ik, 0] = k-1, o
            if k == 2:
                numer = float(((r-l)**2)*(k-1)*(k-1)*(k-1))
                denom = float(((2*(k-1))**2)*(2*(k-1)+1))
            else:
                numer = float(((r-l)**2)*(k-1)*(k-1)*(k-1)*(k-1))
                denom = float(((2*(k-1))**2)*(2*(k-1)+1)*(2*(k-1)-1))
            ab[ik, 1] = numer / denom

    elif typestr == 'hermite' or typestr == 'gaussian':
        mu = 0
        mu0 = math.gamma(mu+0.5)
        if n==1:
            ab = np.array([[0, mu0]])
        else:
            ab = np.zeros([n, 2])
            nvechalf = np.array(range(1, n))*0.5
            nvechalf[0::2] += mu
            ab[0, 1], ab[1::, 1] = mu0, nvechalf

    # Define J, the jacobi matrix from recurrence coefficients in ab
    J = np.zeros([n, n], float)
    if n == 1:
         J = np.array([[ab[0, 0]]])
    else:
        J[0, 0] = ab[0, 0]
        J[0, 1] = math.sqrt(ab[1, 1])
        for i in np.arange(2, n, 1):
            ii = i-1
            J[ii, ii] = ab[ii,0]
            J[ii, ii-1] = math.sqrt(ab[ii, 1])
            J[ii, ii+1] = math.sqrt(ab[ii+1, 1])
        J[n-1, n-1] = ab[n-1, 0]
        J[n-1, n-2] = math.sqrt(ab[n-1, 1])

    return J

def _makeIter(x):
    try:
        iter(x)
        return [xi for xi in x]
    except:
        return [x]
