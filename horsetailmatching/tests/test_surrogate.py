import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
'../../horsetailmatching/')))

from surrogates import PolySurrogate
from parameters import UncertainParameter
from hm import HorsetailMatching
from demoproblems import TP1, TP2, TP3


class TestSurrogate(unittest.TestCase):

    def testPolySurrogate(self):

        def fqoi(x, u):
            return TP1(x, u, jac=False)

        uparams = [UncertainParameter('uniform'), UncertainParameter('uniform')]

        theHM = HorsetailMatching(fqoi, uparams, n_samples_prob=10)
        theHM.surrogate_jac = False
        ans1 = theHM.evalMetric([0, 1])

        poly = PolySurrogate(dimensions=2, order=2)
        theHM.surrogate = poly.surrogate
        theHM.u_surrogate_points = poly.getQuadraturePoints()
        ans2 = theHM.evalMetric([0, 1])

        poly = PolySurrogate(dimensions=2, order=3)
        theHM.surrogate = poly.surrogate
        theHM.u_surrogate_points = poly.getQuadraturePoints()
        ans3 = theHM.evalMetric([0, 1])

        poly = PolySurrogate(dimensions=2, order=4)
        theHM.surrogate = poly.surrogate
        theHM.u_surrogate_points = poly.getQuadraturePoints()
        ans4 = theHM.evalMetric([0, 1])

        self.assertAlmostEqual(ans1, ans2)
        self.assertAlmostEqual(ans2, ans3)
        self.assertAlmostEqual(ans3, ans4)

        theHM = HorsetailMatching(fqoi, uparams,
                surrogate=poly.surrogate, surrogate_jac=True,
                u_surrogate_points=poly.getQuadraturePoints())

    def testPolySurrogateGrad(self):

        def fqoi(x, u): return TP1(x, u, jac=False)

        def fgrad(x, u): return TP1(x, u, jac=True)[1]

        def fboth(x, u): return TP1(x, u, jac=True)

        def ftarget_u(h): return 0 - h**5

        def ftarget_l(h): return -1 - h**5

        u_1 = UncertainParameter('uniform', lower_bound=-1, upper_bound=1)
        u_2 = UncertainParameter('gaussian', mean=0, standard_deviation=1)
        u_3 = UncertainParameter('interval', lower_bound=-1, upper_bound=1)

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

        theHM = HorsetailMatching(fboth, [u_1, u_2, u_3], jac=True,
                  ftarget=(ftarget_u, ftarget_l),
                  n_samples_prob=3, n_samples_int=2,
                  q_integration_points=np.linspace(0, 10, 500),
                  kernel_bandwidth=0.01)
        ans1, grad1 = theHM.evalMetric([0, 1])

        theHM.surrogate = mySurrogateWithGrad
        theHM.surrogate_jac = True
        theHM.u_surrogate_points = u_quad_points
        ans2, grad2 = theHM.evalMetric([0, 1])

        self.assertAlmostEqual(ans1, ans2)
        self.assertAlmostEqual(grad1[0], grad2[0])
        self.assertAlmostEqual(grad1[1], grad2[1])

        qPolyChaos = PolySurrogate(dimensions=3, order=2,
                        poly_type=['legendre', 'hermite', 'legendre'])
        gradPolyChaos = [PolySurrogate(dimensions=3, order=2,
                            poly_type=['legendre', 'hermite', 'legendre']),
                         PolySurrogate(dimensions=3, order=2,
                             poly_type=['legendre', 'hermite', 'legendre'])]
        u_quad_points = qPolyChaos.getQuadraturePoints()

        def mySurrogate(u_quad, q_quad):
            qPolyChaos.train(q_quad)
            return qPolyChaos.predict

        def mySurrogateGrad(u_quad, grad_quad):
            for i, gPC in enumerate(gradPolyChaos):
                gPC.train(grad_quad[:, i])
            def gradmodel(u):
                return [gPC.predict(u) for gPC in gradPolyChaos]
            return gradmodel

        theHM.surrogate = mySurrogate
        theHM.surrogate_jac = mySurrogateGrad
        theHM.u_surrogate_points = u_quad_points
        ans3, grad3 = theHM.evalMetric([0, 1])
        self.assertAlmostEqual(ans1, ans3)
        self.assertAlmostEqual(grad1[0], grad3[0])
        self.assertAlmostEqual(grad1[1], grad3[1])
