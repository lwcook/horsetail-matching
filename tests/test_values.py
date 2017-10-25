import os
import sys
import unittest
import pdb
import copy

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '../horsetailmatching/')))

from hm import HorsetailMatching
from densitymatching import DensityMatching
from parameters import GaussianParameter, UniformParameter, IntervalParameter
from parameters import UncertainParameter
from surrogates import PolySurrogate
from demoproblems import TP0, TP1, TP2, TP3, TP2b


class TestHorsetailMatching(unittest.TestCase):

    def testMetricValues(self):


        ftarget = lambda h: 0
        fqoi = lambda x, u: 1

        uparams = [UniformParameter()]
        theHM = HorsetailMatching(fqoi, uparams, ftarget=ftarget,
                method='kernel',
                integration_points=np.linspace(0.5, 1.5, 200))
        ans = theHM.evalMetric([0])
        self.assertAlmostEqual(ans, np.sqrt(2), places=3)

        theHM.kernel_type = 'uniform'
        ans = theHM.evalMetric([0])
        self.assertAlmostEqual(ans, np.sqrt(2), places=3)

        theHM.kernel_type = 'triangle'
        ans = theHM.evalMetric([0])
        self.assertAlmostEqual(ans, np.sqrt(2), places=3)

        ans = theHM.evalMetric([0], method='empirical')
        self.assertAlmostEqual(ans, np.sqrt(2), places=3)

        ftarget = lambda h: -h
        fqoi = lambda x, u: np.linalg.norm(u)

        up = [UniformParameter(), IntervalParameter()]
        theHM = HorsetailMatching(fqoi, up, ftarget=(ftarget, ftarget),
                samples_prob=100, samples_int=50)
        ans = theHM.evalMetric([0])
        self.assertTrue(abs(ans - 2.05) < 5e-2)

        ftarget = lambda h: -h
        fqoi = lambda x, u: u
        up = UniformParameter()
        theHM = HorsetailMatching(fqoi, up, ftarget=ftarget,
                samples_prob=1000)
        ans = theHM.evalMetric([0])
        self.assertTrue(abs(ans - 1.4) < 1e-1)

        up = IntervalParameter()
        theHM.samples_int=50
        theHM.uncertain_parameters = up
        print(theHM.evalMetric([1]))

    def testGradients(self):

        def findiff(f, x0):
            f0 = f(x0)
            eps = 1e-7
            try:
                iter(x0)
                g = []
                for ix, xi in enumerate(x0):
                    x = copy.copy(x0)
                    x[ix] += eps
                    fnew = f(x)
                    g.append(float(fnew - f0)/eps)
            except TypeError:
                g = float((f(x0 + eps) - f0)/eps)
            return g

        xa = [1, 1]
        xb = [2, 3]
        ftarg = lambda h: 0

        uparams = [UniformParameter(), IntervalParameter()]

        fq = lambda x, u: TP1(x, u, jac=True)
        theHM = HorsetailMatching(fq, uparams, jac=True, ftarget=ftarg,
                samples_prob=20,samples_int=5,
                integration_points=np.linspace(-10, 20, 100))
        q, grad = theHM.evalMetric(xa)
        gfd = findiff(lambda x: theHM.evalMetric(x)[0], xa)
        error1a = np.linalg.norm(np.array(grad) - np.array(gfd))

        q, grad = theHM.evalMetric(xb)
        gfd = findiff(lambda x: theHM.evalMetric(x)[0], xb)
        error1b = np.linalg.norm(np.array(grad) - np.array(gfd))

        fq = lambda x, u: TP2(x, u, jac=True)
        theHM = HorsetailMatching(fq, uparams, jac=True, ftarget=ftarg,
                samples_prob=20, samples_int=5,
                integration_points=np.linspace(-10, 20, 100))
        q, grad = theHM.evalMetric(xa)
        gfd = findiff(lambda x: theHM.evalMetric(x)[0], xa)
        error2a = np.linalg.norm(np.array(grad) - np.array(gfd))

        q, grad = theHM.evalMetric(xb)
        gfd = findiff(lambda x: theHM.evalMetric(x)[0], xb)
        error2b = np.linalg.norm(np.array(grad) - np.array(gfd))

        self.assertAlmostEqual(error1a, 0., places=3)
        self.assertAlmostEqual(error1b, 0., places=3)

        self.assertAlmostEqual(error1a, 0., places=3)
        self.assertAlmostEqual(error1b, 0., places=3)

class TestDensityMatching(unittest.TestCase):


    def testMetricValues(self):

        fqoi = lambda x, u: u
        def ftarg(q):
            if q < -1 or q > 1:
                return 0
            else:
                return 0.5

        uparams = [UniformParameter()]
        theHM = DensityMatching(fqoi, uparams, ftarget=ftarg,
                samples_prob=5000,
                integration_points=np.linspace(-1.2, 1.2, 500))
        ans = theHM.evalMetric([0])
        self.assertAlmostEqual(ans, 0, places=1)

        def ftarg(q):
            if q < -0.5 or q > 0.5:
                return 0
            else:
                return 1

        uparams = [UniformParameter()]
        theHM = DensityMatching(fqoi, uparams, ftarget=ftarg,
                samples_prob=10000,
                integration_points=np.linspace(-1.2, 1.2, 1000))
        ans = theHM.evalMetric([0])
        self.assertAlmostEqual(ans, 0.5, places=1)

    def testGradients(self):

        def findiff(f, x0):
            f0 = f(x0)
            eps = 1e-7
            try:
                iter(x0)
                g = []
                for ix, xi in enumerate(x0):
                    x = copy.copy(x0)
                    x[ix] += eps
                    fnew = f(x)
                    g.append(float(fnew - f0)/eps)
            except TypeError:
                g = float((f(x0 + eps) - f0)/eps)
            return g

        xa = [1, 1]
        xb = [2, 3]
        def ftarg(q):
            if q < 0 or q > 10:
                return 0
            else:
                return 0.1

        uparams = [UniformParameter(), GaussianParameter()]

        fq = lambda x, u: TP1(x, u, jac=True)
        theDM = DensityMatching(fq, uparams, jac=True, ftarget=ftarg,
                samples_prob=200,
                integration_points=np.linspace(-10, 20, 100))
        q, grad = theDM.evalMetric(xa)
        gfd = findiff(lambda x: theDM.evalMetric(x)[0], xa)
        error1a = np.linalg.norm(np.array(grad) - np.array(gfd))

        q, grad = theDM.evalMetric(xb)
        gfd = findiff(lambda x: theDM.evalMetric(x)[0], xb)
        error1b = np.linalg.norm(np.array(grad) - np.array(gfd))

        fq = lambda x, u: TP2(x, u, jac=True)
        theDM = DensityMatching(fq, uparams, jac=True, ftarget=ftarg,
                samples_prob=200,
                integration_points=np.linspace(-10, 20, 100))
        q, grad = theDM.evalMetric(xa)
        gfd = findiff(lambda x: theDM.evalMetric(x)[0], xa)
        error2a = np.linalg.norm(np.array(grad) - np.array(gfd))

        q, grad = theDM.evalMetric(xb)
        gfd = findiff(lambda x: theDM.evalMetric(x)[0], xb)
        print grad
        print gfd
        error2b = np.linalg.norm(np.array(grad) - np.array(gfd))

        self.assertAlmostEqual(error1a, 0., places=3)
        self.assertAlmostEqual(error1b, 0., places=3)

        self.assertAlmostEqual(error1a, 0., places=3)
        self.assertAlmostEqual(error1b, 0., places=3)

if __name__ == "__main__":
    unittest.main()
