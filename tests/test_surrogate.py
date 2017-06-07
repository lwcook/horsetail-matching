import unittest
import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '../horsetailmatching/')))

from surrogates import PolySurrogate
from parameters import UncertainParameter
from hm import HorsetailMatching
from demoproblems import TP0, TP1, TP2, TP3


class TestSurrogate(unittest.TestCase):

    def testPolySurrogate(self):

        def fqoi(x, u):
            return TP1([x, 0], [u, 0])

        uparam = UncertainParameter('uniform')

        poly1 = PolySurrogate(dimensions=1, order=0, poly_type='hermite')
        theHM = HorsetailMatching(fqoi, uparam, samples_prob=10,
            surrogate=poly1.surrogate,
            surrogate_points=poly1.getQuadraturePoints())
        ans1 = theHM.evalMetric(1)

        poly2 = PolySurrogate(dimensions=1, order=0, poly_type=['legendre'])
        theHM.surrogate=poly2.surrogate
        theHM.surrogate_points=poly2.getQuadraturePoints()
        ans2 = theHM.evalMetric(1)
        self.assertAlmostEqual(poly1.coeffs[0], poly2.coeffs[0])

        poly1.predict(1)
        poly2.predict(1)

        poly2 = PolySurrogate(dimensions=1, order=0, poly_type=['legendre'])
        theHM = HorsetailMatching(fqoi, uparam, samples_prob=10,
                surrogate=poly2.surrogate)
        theHM.evalMetric(1)

        def fqoi(x, u):
            return TP1(x, u, jac=False)

        uparams = [UncertainParameter('uniform'), UncertainParameter('uniform')]
        poly = PolySurrogate(dimensions=2, order=3)

        theHM = HorsetailMatching(fqoi, uparams, samples_prob=10)
        ansTrue = theHM.evalMetric([0, 1])

        theHM.surrogate = poly.surrogate
        theHM.surrogate_jac = False
        theHM.surrogate_points = poly.getQuadraturePoints()
        theHM.evalMetric([0, 1])
        ans1 = theHM.evalMetric([0, 1])

        self.assertAlmostEqual(ansTrue, ans1)

        theHM.surrogate = poly.surrogate
        theHM.surrogate_points = poly.getQuadraturePoints()
        ans2 = theHM.evalMetric([0, 1])

        poly = PolySurrogate(dimensions=2, order=3)
        theHM.surrogate = poly.surrogate
        theHM.surrogate_points = poly.getQuadraturePoints()
        ans3 = theHM.evalMetric([0, 1])

        poly = PolySurrogate(dimensions=2, order=4)
        theHM.surrogate = poly.surrogate
        theHM.surrogate_points = poly.getQuadraturePoints()
        ans4 = theHM.evalMetric([0, 1])

        self.assertAlmostEqual(ans1, ans2)
        self.assertAlmostEqual(ans2, ans3)
        self.assertAlmostEqual(ans3, ans4)

        theHM = HorsetailMatching(fqoi, uparams,
                surrogate=poly.surrogate, surrogate_jac=True,
                surrogate_points=poly.getQuadraturePoints())

    def testPolySurrogateGrad(self):

        def fqoi(x, u): return TP1(x, u, jac=False)

        def fgrad(x, u): return TP1(x, u, jac=True)[1]

        def fboth(x, u): return TP1(x, u, jac=True)

        def ftarget_u(h): return 0 - h**5

        def ftarget_l(h): return -1 - h**5

        u_1 = UncertainParameter('uniform', lower_bound=-1, upper_bound=1)
        u_2 = UncertainParameter('gaussian', mean=0, standard_deviation=1)
        u_3 = UncertainParameter('interval', lower_bound=-1, upper_bound=1)

        qPolyChaos = PolySurrogate(dimensions=3, order=2,
                        poly_type=['legendre', 'hermite', 'legendre'])

        gradPolyChaos = [PolySurrogate(dimensions=3, order=2,
                            poly_type=['legendre', 'hermite', 'legendre']),
                         PolySurrogate(dimensions=3, order=2,
                             poly_type=['legendre', 'hermite', 'legendre'])]

        u_quad_points = qPolyChaos.getQuadraturePoints()

        def mySurrogate(u_quad, q_quad):
            qPolyChaos = PolySurrogate(dimensions=3, order=2,
                            poly_type=['legendre', 'hermite', 'legendre'])
            qPolyChaos.train(q_quad)
            return qPolyChaos.predict

        def mySurrogateGrad(u_quad, grad_quad):
            for i, gPC in enumerate(gradPolyChaos):
                gPC.train(grad_quad[:, i])
            def gradmodel(u):
                return [gPC.predict(u) for gPC in gradPolyChaos]
            return gradmodel

        def mySurrogateWithGrad(u_quad, q_quad, grad_quad):
            qPolyChaos.train(q_quad)
            for i, gPC in enumerate(gradPolyChaos):
                gPC.train(grad_quad[:, i])
            def qmodel(u):
                return qPolyChaos.predict(u)
            def gradmodel(u):
                return [gPC.predict(u) for gPC in gradPolyChaos]
            return qmodel, gradmodel

        theHM = HorsetailMatching(fqoi, [u_1, u_2, u_3], jac=fgrad,
                  ftarget=(ftarget_u, ftarget_l),
                  samples_prob=3, samples_int=2,
                  integration_points=np.linspace(0, 5, 10),
                  kernel_bandwidth=0.01)
        ansTrue, gradTrue = theHM.evalMetric([0, 1])

        theHM.surrogate = mySurrogateWithGrad
        theHM.surrogate_jac = True
        theHM.surrogate_points = u_quad_points
        ans0, grad0 = theHM.evalMetric([0, 1])

        self.assertAlmostEqual(ansTrue, ans0)
        self.assertAlmostEqual(gradTrue[0], grad0[0])
        self.assertAlmostEqual(gradTrue[1], grad0[1])

        theHM.fqoi = fboth
        theHM.jac = True
        ans1, grad1 = theHM.evalMetric([0, 1])

        self.assertAlmostEqual(ans1, ans0)
        self.assertAlmostEqual(grad1[0], grad0[0])
        self.assertAlmostEqual(grad1[1], grad0[1])

        theHM.surrogate = mySurrogateWithGrad
        theHM.surrogate_jac = True
        ans2, grad2 = theHM.evalMetric([0, 1])

        self.assertAlmostEqual(ans1, ans2)
        self.assertAlmostEqual(grad1[0], grad2[0])
        self.assertAlmostEqual(grad1[1], grad2[1])

        theHM.surrogate = mySurrogate
        theHM.surrogate_jac = False
        ans3, grad3 = theHM.evalMetric([0, 1])

        self.assertAlmostEqual(ans2, ans3)
        self.assertAlmostEqual(grad2[0], grad3[0])
        self.assertAlmostEqual(grad2[1], grad3[1])

        theHM.surrogate = mySurrogate
        theHM.surrogate_jac = mySurrogateGrad
        ans4, grad4 = theHM.evalMetric([0, 1])

        self.assertAlmostEqual(ans4, ans1)
        self.assertAlmostEqual(grad4[0], grad1[0])
        self.assertAlmostEqual(grad4[1], grad1[1])
