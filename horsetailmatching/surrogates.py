import numpy as np
import pdb

def polySurrogate(x_sp, f_sp, dims=None):
    """ Creates a simpler polynomial surrogate to
    model a function over the range [-1,1]**dims """

    if dims is None:
        dims = x_sp.shape[1]

    if dims == 1:
        X, F = x_sp.flatten(), f_sp.flatten()

        A = np.array([X * 0. + 1., X, X**2, X**3, X**4, X**5]).T
        c, r, rank, s = np.linalg.lstsq(A, F)

        def poly_model(u):
            return c[0] + c[1]*u + c[2]*u**2 + c[3]*u**3 +\
                c[4]*u**4 + c[5]*u**5

        return poly_model

    elif dims == 2:

        X, Y = x_sp[:, 0].flatten(), x_sp[:, 1].flatten()
        F = f_sp.flatten()

        A = np.array(
            [X*0+1., X, Y, X**2, X*Y, Y**2, X**3, X**2*Y, X*Y**2, Y**3]).T
        c, r, rank, s = np.linalg.lstsq(A, F)

        def poly_model(u):
            x = u[0]
            y = u[1]
            return c[0] + c[1]*x + c[2]*y + \
                c[3]*x**2 + c[4]*x*y + c[5]*y**2 + \
                c[6]*x**3 + c[7]*x**2*y + c[8]*x*y**2 + c[9]*y**3

        return poly_model

    elif dims == 3:

        X = x_sp[:, 0].flatten()
        Y = x_sp[:, 1].flatten()
        Z = x_sp[:, 2].flatten()
        F = f_sp.flatten()

        A = np.array(
            [X*0+1.,
             X, Y, Z,
             X**2, Y**2, Z**2,
             X*Y, X*Z, Y*Z,
             X**3, Y**3, Z**3,
             X**2*Y, X**2*Z, Y**2*X,
             Y**2*Z, Z**2*X, Z**2*Y,
             X*Y*Z
            ]).T

        c, r, rank, s = np.linalg.lstsq(A, F)

        def poly_model(u):
            x = u[0]
            y = u[1]
            z = u[2]
            return c[0] + \
                c[1]*x + c[2]*y + c[3]*z + \
                c[4]*x**2 + c[5]*y**2 + c[6]*z**2 + \
                c[7]*x*y + c[8]*x*z + c[9]*y*z + \
                c[10]*x**3 + c[11]*y**3 + c[12]*y**3 + \
                c[13]*x**2*y + c[14]*x**2*z + c[15]*y**2*z + \
                c[16]*y**2*x + c[17]*z**2*x + c[18]*z**2*y + \
                c[19]*x*y*z

        return poly_model

class NIPC(object):

    def __init__(self, dimensions=[], order=3, poly_type='legendre'):

        self.dims = dimensions
        self.P = int(order) + 1

        if isinstance(poly_type, basestring):
            self.poly_types = [poly_type for _ in np.arange(self.dims)]
        else:
            self.poly_types = utils.makeIter(poly_type)
        self.J_list = [_define_poly_J(p, self.P) for p in self.poly_types]

        imesh = np.meshgrid(*[np.arange(self.P) for d in np.arange(self.dims)])
        self.index_polys = np.vstack([m.flatten() for m in imesh]).T 
        self.N_poly = len(self.index_polys)
        self.coeffs = np.zeros([self.P for __ in np.arange(self.dims)])

    def surrogate(u_sparse, q_sparse):
        '''Returns a surrogate model function fitted to the input/output
        combinations given in u_sparse and q_sparse.

            u_sparse: input values at which the output values are obtained.
                Must be the same as the qaudrature points defined by the
                getQuadraturePoints method.
            q_sparse: output values corresponding to the input values given in
                u_sparse to which the surrogate is fitted
        '''
        self.train(q_sparse)
        def model(u):
            return self.predict(u)
        return model

    def predict(self, u):
        '''Predicts the output value at u from the fitted polynomial expansion.
        Therefore the method train() must be called first.

            u: input value at which to predict the output.
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

            fpoints: output values corresponding to the quadrature points given
            by the getQuadraturePoints method to which the expansion should be
            trained.
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

        returns:
            upoints: array of size (num_polynomials, num_dimensions)
            wpoints: array of size (num_polynomials)
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

        returns:
            upoints: array of size (num_polynomials, num_dimensions)
        '''
        upoints, _ = self.getQuadraturePointsAndWeights()
        return upoints

## --------------------------------------------------------------------------
## Private funtions for polynomials
## --------------------------------------------------------------------------

def eval_poly(us, ns, Js):
    '''Evaluate multi-dimensional polynomials through tensor multiplication.
        us: vector value at which to evaluate the polynomial
        ns: order in each dimension at which to evaluate the polynomial
        Js: Jacobi matrix of each dimension's 1D polynomial
        '''
    us = utils.makeIter(us)
    ns = utils.makeIter(ns)
    Js = utils.makeIter(Js)
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
