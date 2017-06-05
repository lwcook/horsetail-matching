import numpy as np
import scipy.optimize as scopt
import matplotlib.pyplot as plt

from horsetailmatching import HorsetailMatching, UncertainParameter
from horsetailmatching.demoproblems import TP1
from horsetailmatching.surrogates import PolySurrogate

u_1 = UncertainParameter('uniform', lower_bound=-1, upper_bound=1)
u_2 = UncertainParameter('gaussian', mean=0, standard_deviation=1)
u_3 = UncertainParameter('interval', lower_bound=-1, upper_bound=1)

def fQOI(x, u):
    return TP1(x, u, jac=True)

def ftarget_u(h):
    return 0 - h**5

def ftarget_l(h):
    return -1 - h**5

qPolyChaos = PolySurrogate(dimensions=3, order=3,
        poly_type=['legendre', 'hermite', 'legendre'])

gradPolyChaos = [PolySurrogate(dimensions=3, order=3,
                    poly_type=['legendre', 'hermite', 'legendre']),
                 PolySurrogate(dimensions=3, order=3,
                     poly_type=['legendre', 'hermite', 'legendre'])]

u_quad_points = qPolyChaos.getQuadraturePoints()

def mySurrogateWithGrad(u_quad, q_quad, grad_quad):
    qPolyChaos.train(q_quad)
    for i, gPC in enumerate(gradPolyChaos):
        gPC.train(grad_quad[:, i])
    def qmodel(u):
        return qPolyChaos.predict(u)
    def gradmodel(u):
        return [gPC.predict(u) for gPC in gradPolyChaos]
    return qmodel, gradmodel

theHM = HorsetailMatching(fQOI, [u_1, u_2, u_3], jac=True,
          ftarget_u=ftarget_u, ftarget_l=ftarget_l,
          n_samples_prob=3, n_samples_int=2,
          q_integration_points=np.linspace(0, 10, 500),
          surrogate=mySurrogateWithGrad, surrogate_jac=True,
          u_quadrature_points=u_quad_points, bw=0.01,
          verbose=True)

def myObj(x):
    q, grad = theHM.evalMetric(x)
#    theHM.plotHorsetail()
#    plt.show()
    return q, grad

solution = scopt.minimize(myObj, x0=[3, 2], jac=True, method='BFGS')
print solution
theHM.plotHorsetail()
plt.show()
