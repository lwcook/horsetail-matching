import numpy as np

def TP0(dv, u):
    '''Demo problem 0 for horsetail matching, takes two input vectors of any size
    and returns a single output'''
    return np.linalg.norm(np.array(dv)) + np.linalg.norm(np.array(u))

def TP1(x, u, jac=False):
    '''Demo problem 1 for horsetail matching, takes two input vectors of size 2
    and returns just the qoi if jac is False or the qoi and its gradient if jac
    is True'''
    factor = 0.1*(u[0]**2 + 2*u[0]*u[1] + u[1]**2)
    q = 0 + factor*(x[0]**2 + 2*x[1]*x[0] + x[1]**2)
    if not jac:
        return q
    else:
        grad = [factor*(2*x[0] + x[1]), factor*(x[0] + 2*x[1])]
        return q, grad

def TP2(dv, u, jac=False):
    '''Demo problem 2 for horsetail matching, takes two input vectors of size 2
    and returns just the qoi if jac is False or the qoi and its gradient if jac
    is True'''
    y = dv[0]/2.
    z = dv[1]/2. + 12

    q = 0.25*((y**2 + z**2)/10 + 5*u[0]*u[1] - z*u[1]**2) + 0.2*z*u[1]**3 + 7

    if not jac:
        return q
    else:
        dqdx1 = (1./8.)*( (2*y)/40. + 5*u[0]*u[1])
        dqdx2 = (1./8.)*( (2*z)/40. - u[1]**2) + 0.1*u[1]**3
        return q, [dqdx1, dqdx2]

def TP3(x, u, jac=False):
    '''Demo problem 1 for horsetail matching, takes two input values of
    size 1'''

    q = 2 + 0.5*x + 1.5*(1-x)*u
    if not jac:
        return q
    else:
        grad = 1 - 0.9*u
        return q, grad
