import os
import sys
import unittest
import pdb
import copy
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '../horsetailmatching/')))

from hm import HorsetailMatching
from densitymatching import DensityMatching
from weightedsum import WeightedSum
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

        theHM = HorsetailMatching(TP0, GaussianParameter(),
                integration_points=np.linspace(-1, 100, 100))

        with self.assertRaises(ValueError):
            theHM.getHorsetail()

        theHM.evalMetric([0, 1])
        theHM = HorsetailMatching(TP0, [], IntervalParameter())
        theHM.evalMetric([0, 1])
        theHM = HorsetailMatching(TP0, UniformParameter())
        theHM.evalMetric([0, 1])
        theHM = HorsetailMatching(TP0, [GaussianParameter()])
        theHM.evalMetric([0, 1])

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

        theHM = HorsetailMatching(fqoi, UniformParameter(), IntervalParameter(),
                ftarget=ftarget)
        theHM.evalMetric([1, 1], method='empirical')
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()

        theHM.evalMetric([1, 1], method='kernel')
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()

        theHM = HorsetailMatching(fqoi, UniformParameter(), IntervalParameter(),
                ftarget=ftarget,
                samples_prob=5, samples_int=3)

        theHM.evalMetric([1, 1], method='kernel')
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()

        ups = [UniformParameter(), GaussianParameter()]
        theHM.prob_uncertainties = ups
        theHM.int_uncertainties = IntervalParameter()

        theHM.evalMetric([1, 1], method='kernel')
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()
        theHM.evalMetric([1, 1], method='empirical')
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()

        uparams = [IntervalParameter(), IntervalParameter()]

        theHM = HorsetailMatching(fqoi, uparams,
                verbose=True, reuse_samples=True)

        theHM.prob_uncertainties = []
        theHM.int_uncertainties = uparams

        theHM.evalMetric([1,1], method='kernel')
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()
        theHM.evalMetric([1,1], method='empirical')
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()

        theHM = HorsetailMatching(fqoi, [], uparams)
        theHM = HorsetailMatching(fqoi, [], uparams, ftarget=ftarget)
        theHM = HorsetailMatching(fqoi, [], uparams, ftarget=(ftarget, ftarget))
        theHM = HorsetailMatching(fqoi, [], uparams, samples_prob=100,
                samples_int = 50)

        theHM = HorsetailMatching(fqoi, [], uparams, method='kernel',
                integration_points=np.linspace(-50, 50, 1000),
                kernel_bandwidth=0.01)

        theHM = HorsetailMatching(fboth, UniformParameter(), IntervalParameter(),
                jac=True, samples_prob=5,
                integration_points=np.linspace(-50, 50, 1000),
                samples_int=3, verbose=True, reuse_samples=True)

        theHM.evalMetric([1, 1], method='kernel')
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()
        theHM.evalMetric([1, 1], method='empirical')
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()

        theHM.fqoi = fqoi
        theHM.jac = fgrad
        theHM.evalMetric([1, 1], method='kernel')
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()
        theHM.evalMetric([1, 1], method='empirical')
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()

        theHM.prob_uncertainties = UniformParameter()
        theHM.int_uncertainties = []
        theHM.fqoi=TP0
        theHM.jac=False
        theHM.evalMetric([1, 1])
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()

        theHM.reuse_samples = False
        theHM.evalMetric([1, 1])
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()

        theHM = HorsetailMatching(fboth, UniformParameter(), IntervalParameter(),
                jac=True, samples_prob=5,
                samples_int=3, kernel_bandwidth=0.01, kernel_type='uniform')
        theHM.evalMetric([1, 1])
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()

        theHM = HorsetailMatching(fboth, UniformParameter(), IntervalParameter(),
                jac=True, samples_prob=5,
                samples_int=3, kernel_bandwidth=0.01, kernel_type='triangle')
        theHM.evalMetric([1, 1])
        (x1, y1, t1), (x2, y2, t2), CDFs = theHM.getHorsetail()

        with self.assertRaises(ValueError):
            theHM = HorsetailMatching(fboth, UniformParameter(), IntervalParameter(),
                    jac=True, samples_prob=5,
                    samples_int=3, kernel_bandwidth=0.01, kernel_type='bad')

        theHM._getParameterSamples()

    def testDM(self):

        theDM = DensityMatching(TP0, GaussianParameter())
        with self.assertRaises(ValueError):
            theDM.getPDF()

        theDM.evalMetric([0, 1])
        theDM = DensityMatching(TP0, UniformParameter())
        theDM.evalMetric([0, 1])
        theDM = DensityMatching(TP0, [GaussianParameter()])
        theDM.evalMetric([0, 1])

        def fqoi(x, u):
            return TP1(x, u, jac=False)

        def fgrad(x, u):
            return TP1(x, u, jac=True)[1]

        def fboth(x, u):
            return TP1(x, u, jac=True)

        def ftarget(q):
            if q < 0 or q > 5:
                return 0
            else:
                return 0.2

        def fzero(x, u):
            return 0

        uparams = [UniformParameter(), UniformParameter()]

        theDM = DensityMatching(fzero, uparams, ftarget=ftarget, verbose=True)
        theDM.evalMetric([1, 1])

        theDM = DensityMatching(fqoi, uparams, ftarget=ftarget, verbose=True)
        theDM.evalMetric([1, 1])
        (x1, y1, t1) = theDM.getPDF()

        theDM = DensityMatching(fboth, uparams, jac=True, ftarget=ftarget, verbose=True)
        theDM.evalMetric([1, 1])

        theDM = DensityMatching(fqoi, uparams, jac=fgrad, ftarget=ftarget, verbose=True)
        theDM.evalMetric([1, 1])

        _ = theDM.ftarget

        theDM.u_samples = None
        with self.assertRaises(TypeError):
            theDM.u_samples = np.array([0])
        with self.assertRaises(TypeError):
            theDM.u_samples = 1

    def testWS(self):

        theWS = WeightedSum(TP0, GaussianParameter())

        theWS.evalMetric([0, 1])
        theWS = WeightedSum(TP0, UniformParameter())
        theWS.evalMetric([0, 1])
        theWS = WeightedSum(TP0, [GaussianParameter()])
        theWS.evalMetric([0, 1])

        def fqoi(x, u):
            return TP1(x, u, jac=False)

        def fgrad(x, u):
            return TP1(x, u, jac=True)[1]

        def fboth(x, u):
            return TP1(x, u, jac=True)

        def fzero(x, u):
            return 0

        uparams = [UniformParameter(), UniformParameter()]

        theWS = WeightedSum(fzero, uparams, verbose=True)
        theWS.evalMetric([1, 1])

        theWS = WeightedSum(fqoi, uparams, verbose=True)
        theWS.evalMetric([1, 1])

        theWS = WeightedSum(fboth, uparams, jac=True, verbose=True)
        theWS.evalMetric([1, 1])

        theWS = WeightedSum(fqoi, uparams, jac=fgrad, verbose=True)
        theWS.evalMetric([1, 1])

        theWS.u_samples = None
        with self.assertRaises(TypeError):
            theWS.u_samples = np.array([0])
        with self.assertRaises(TypeError):
            theWS.u_samples = 1


if __name__ == "__main__":
    unittest.main()
