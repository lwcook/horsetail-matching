import os
import sys
import unittest
import pdb
import copy

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '../horsetailmatching/')))

from hm import HorsetailMatching
from parameters import GaussianParameter, UniformParameter, IntervalParameter
from parameters import UncertainParameter
from surrogates import PolySurrogate
from demoproblems import TP0, TP1, TP2, TP3, TP2b

class TestInitializations(unittest.TestCase):

    def testDemoProblems(self):

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

        x0 = [1, 1]
        u0 = [0.5, 0.5]

        ans = TP0(x0, u0)
        ans = TP1(x0, u0)
        ans = TP2(x0, u0)
        ans = TP2b(x0, u0)
        ans = TP3(1, 1)

        q, grad = TP1(x0, u0, jac=True)
        f = lambda x: TP1(x, u0)
        gfd = findiff(f, x0)
        error1 = np.linalg.norm(np.array(grad) - np.array(gfd))

        q, grad = TP2(x0, u0, jac=True)
        f = lambda x: TP2(x, u0)
        gfd = findiff(f, x0)
        error2 = np.linalg.norm(np.array(grad) - np.array(gfd))

        q, grad = TP2b(x0, u0, jac=True)
        f = lambda x: TP2b(x, u0)
        gfd = findiff(f, x0)
        error2b = np.linalg.norm(np.array(grad) - np.array(gfd))

        q, grad = TP3(1, 1, jac=True)
        f = lambda x: TP3(x, 1)
        gfd = findiff(f, 1)
        error3 = np.linalg.norm(np.array(grad) - np.array(gfd))

        self.assertAlmostEqual(error1, 0., places=5)
        self.assertAlmostEqual(error2, 0., places=5)
        self.assertAlmostEqual(error2b, 0., places=5)
        self.assertAlmostEqual(error3, 0., places=5)


    def testUncertainParameter(self):

        param = GaussianParameter()
        param.getSample()
        param.evalPDF(0)

        param = UniformParameter()
        param.getSample()
        param.evalPDF(0)

        with self.assertRaises(ValueError):
            param.upper_bound = -2

        with self.assertRaises(ValueError):
            param.lower_bound = 3

        param = UniformParameter(lower_bound=-1, upper_bound=1)
        param.getSample()
        param.evalPDF(0)
        self.assertAlmostEqual(param.lower_bound, -1)
        self.assertAlmostEqual(param.upper_bound, 1)
        self.assertEqual(param.evalPDF(-100), 0)

        param = GaussianParameter(mean=1, standard_deviation=3)
        param.getSample()
        param.evalPDF(0)
        self.assertAlmostEqual(param.mean, 1.)
        self.assertAlmostEqual(param.standard_deviation, 3)

        def fpdf(x):
            if x < -1 or x > 1:
                return 0
            else:
                return 1./2.
        param = UncertainParameter(pdf=fpdf, lower_bound=-1, upper_bound=1)
        param.getSample()
        param.evalPDF(0)
        self.assertEqual(param.evalPDF(-100), 0)

        param = IntervalParameter(lower_bound=-2, upper_bound=2)
        param.getSample()
        param.evalPDF(0)
        param.evalPDF(np.array([0]))
        param.evalPDF([0, 1])
        self.assertAlmostEqual(param.lower_bound, -2)
        self.assertAlmostEqual(param.upper_bound, 2)
        self.assertEqual(param.evalPDF(-100), 0)


    def testHM(self):

        theHM = HorsetailMatching(TP0, UniformParameter())
        with self.assertRaises(ValueError):
            theHM.getHorsetail()

        theHM.evalMetric([0, 1])
        theHM = HorsetailMatching(TP0, IntervalParameter())
        theHM.evalMetric([0, 1])
        theHM = HorsetailMatching(TP0, GaussianParameter())
        theHM.evalMetric([0, 1])
        theHM = HorsetailMatching(TP0, [GaussianParameter()])
        theHM.evalMetric([0, 1])

        _ = theHM.uncertain_parameters
        with self.assertRaises(ValueError):
            theHM.uncertain_parameters = []

        _ = theHM.ftarget

        theHM.u_samples = None
        with self.assertRaises(TypeError):
            theHM.u_samples = np.array([0])
        with self.assertRaises(TypeError):
            theHM.u_samples = 1

        with self.assertRaises(ValueError):
            theHM.evalMetric([0, 1], method='badmethod')

        def fqoi(x, u):
            return TP1(x, u, jac=False)

        def fgrad(x, u):
            return TP1(x, u, jac=True)[1]

        def fboth(x, u):
            return TP1(x, u, jac=True)

        def ftarget(h):
            return 0

        uparams = [UniformParameter(), UniformParameter()]
        theHM = HorsetailMatching(fqoi, uparams, ftarget=ftarget)
        theHM.evalMetric([0, 0], method='empirical')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        theHM.uncertain_parameters = uparams
        theHM.evalMetric([0, 0], method='kernel')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        uparams = [UniformParameter(), IntervalParameter()]
        theHM = HorsetailMatching(fqoi, uparams, ftarget=ftarget,
                samples_prob=5, samples_int=3)

        theHM.evalMetric([0,0], method='kernel')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        ups = [UniformParameter(), IntervalParameter(), GaussianParameter()]
        theHM.uncertain_parameters = ups

        theHM.evalMetric([0,0], method='kernel')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()
        theHM.evalMetric([0,0], method='empirical')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        theHM = HorsetailMatching(fqoi, uparams,
                verbose=True, reuse_samples=True)

        uparams = [IntervalParameter(), IntervalParameter()]
        theHM.uncertain_parameters = uparams
        theHM.evalMetric([1,1], method='kernel')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()
        theHM.evalMetric([1,1], method='empirical')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        theHM = HorsetailMatching(fqoi, uparams)
        theHM = HorsetailMatching(fqoi, uparams, ftarget=ftarget)
        theHM = HorsetailMatching(fqoi, uparams, ftarget=(ftarget, ftarget))
        theHM = HorsetailMatching(fqoi, uparams, samples_prob=500,
                samples_int = 50)

        theHM = HorsetailMatching(fqoi, uparams, method='kernel',
                integration_points=np.linspace(0, 10, 100),
                kernel_bandwidth=0.01)

        uparams = [UniformParameter(), IntervalParameter()]
        theHM = HorsetailMatching(fboth, uparams, jac=True, samples_prob=5,
                samples_int=3, verbose=True, reuse_samples=True)

        theHM.evalMetric([1, 1], method='kernel')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()
        theHM.evalMetric([1, 1], method='empirical')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        theHM.fqoi = fqoi
        theHM.jac = fgrad
        theHM.evalMetric([1, 1], method='kernel')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()
        theHM.evalMetric([1, 1], method='empirical')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        theHM.uncertain_parameters = UniformParameter()
        theHM.fqoi=TP0
        theHM.jac=False
        theHM.evalMetric([1, 1])
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        theHM.reuse_samples = False
        theHM.evalMetric([1, 1])
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        theHM = HorsetailMatching(fboth, uparams, jac=True, samples_prob=5,
                samples_int=3, kernel_bandwidth=0.01, kernel_type='uniform')
        theHM.evalMetric([1, 1])
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        theHM = HorsetailMatching(fboth, uparams, jac=True, samples_prob=5,
                samples_int=3, kernel_bandwidth=0.01, kernel_type='triangle')
        theHM.evalMetric([1, 1])
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        with self.assertRaises(ValueError):
            theHM = HorsetailMatching(fboth, uparams, jac=True, samples_prob=5,
                    samples_int=3, kernel_bandwidth=0.01, kernel_type='bad')



class TestHorsetailMatching(unittest.TestCase):


    def testMetricValues(self):

        ftarget = lambda h: 0
        fqoi = lambda x, u: 1

        uparams = [UniformParameter()]
        theHM = HorsetailMatching(fqoi, uparams, ftarget=ftarget,
                method='kernel',
                integration_points=np.linspace(0.99, 1.01, 100))
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



if __name__ == "__main__":
    unittest.main()
