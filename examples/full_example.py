import numpy as np
import scipy.optimize as scopt
import matplotlib.pyplot as plt

from horsetailmatching import HorsetailMatching, UncertainParameter
from horsetailmatching.demoproblems import TP2
from horsetailmatching.surrogates import PolySurrogate

def main():

    u_1 = UncertainParameter('uniform', lower_bound=-1, upper_bound=1)
    u_2 = UncertainParameter('interval', lower_bound=-1, upper_bound=1)

    def fQOI(x, u):
        return TP2(x, u, jac=True)

    def ftarget_u(h):
        return 0 - h**5

    def ftarget_l(h):
        return -1 - h**5

    qPolyChaos = PolySurrogate(dimensions=2, order=3,
            poly_type=['legendre', 'hermite', 'legendre'])

    gradPolyChaos = [PolySurrogate(dimensions=2, order=3,
                        poly_type=['legendre', 'hermite', 'legendre']),
                     PolySurrogate(dimensions=2, order=3,
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

    theHM = HorsetailMatching(fQOI, [u_1, u_2], jac=True,
              ftarget=(ftarget_u, ftarget_l),
              samples_prob=1000, samples_int=25,
              integration_points=np.linspace(-10, 25, 500),
              surrogate=mySurrogateWithGrad, surrogate_jac=True,
              surrogate_points=u_quad_points,
              kernel_type='uniform', verbose=True)

    theHM.evalMetric([1, 1])
    upper, lower, CDFs = theHM.getHorsetail()
    for CDF in CDFs:
        plt.plot(CDF[0], CDF[1], 'grey', lw=0.5)
    plt.plot(upper[0], upper[1], 'b')
    plt.plot(lower[0], lower[1], 'b')
    plt.show()

    def myObj(x):
        q, grad = theHM.evalMetric(x)
    #    theHM.plotHorsetail()
    #    plt.show()
        return q, grad

    solution = scopt.minimize(myObj, x0=[1, 1], jac=True, method='SLSQP',
            constraints=[{'type': 'ineq', 'fun': lambda x: x[0]},
                        {'type': 'ineq', 'fun': lambda x: x[1]}])
    print solution

    upper, lower, CDFs = theHM.getHorsetail()
    for CDF in CDFs:
        plt.plot(CDF[0], CDF[1], 'grey', lw=0.5)
    plt.plot(upper[0], upper[1], 'b')
    plt.plot(lower[0], lower[1], 'b')
    plt.show()

if __name__ == "__main__":
    main()
