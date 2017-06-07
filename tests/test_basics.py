import os
import sys
import unittest
import pdb

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    '../horsetailmatching/')))

from hm import HorsetailMatching
from parameters import UncertainParameter
from surrogates import PolySurrogate
from demoproblems import TP0, TP1, TP2, TP3

class TestInitializations(unittest.TestCase):

    def testDemoProblems(self):

        ans = TP0([0, 1], [0, 1])
        ans = TP1([0, 1], [0, 1])
        ans = TP1([0, 1], [0, 1], jac=True)
        ans = TP2([0, 1], [0, 1])
        ans = TP2([0, 1], [0, 1], jac=True)
        ans = TP3(1, 1)
        ans = TP3(1, 1, jac=True)

    def testUncertainParameter(self):

        param = UncertainParameter()
        param.getSample()
        param.evalPDF(0)
        self.assertEqual(param.distribution, 'uniform')

        param = UncertainParameter('uniform')
        param.getSample()
        param.evalPDF(0)
        self.assertEqual(param.distribution, 'uniform')

        with self.assertRaises(ValueError):
            param = UncertainParameter('baddist')

        with self.assertRaises(ValueError):
            param.upper_bound = -2

        with self.assertRaises(ValueError):
            param.lower_bound = 3

        param = UncertainParameter(distribution='uniform', lower_bound=-1,
                upper_bound=1)
        param.getSample()
        param.evalPDF(0)
        self.assertAlmostEqual(param.mean, 0.)
        self.assertAlmostEqual(param.standard_deviation, 1./np.sqrt(3.))

        param = UncertainParameter(distribution='gaussian', mean=0.,
                standard_deviation=3.)
        param.getSample()
        param.evalPDF(0)
        self.assertAlmostEqual(param.mean, 0.)
        self.assertAlmostEqual(param.standard_deviation, 3)

        fpdf = lambda x: 1./2.
        param = UncertainParameter(distribution='custom', pdf=fpdf)
        param.getSample()
        param.evalPDF(0)
        with self.assertRaises(ValueError):
            param = UncertainParameter(distribution='custom')

        param = UncertainParameter('interval', lower_bound=-2, upper_bound=2)
        param.getSample()
        param.evalPDF(0)
        param.evalPDF(np.array([0]))
        param.evalPDF([0, 1])
        self.assertAlmostEqual(param.lower_bound, -2)
        self.assertAlmostEqual(param.upper_bound, 2)
        self.assertEqual(param.evalPDF(-100), 0)

        def myPDF(q): return 1/(2.5 - 1.5)
        param = UncertainParameter('custom', pdf=myPDF, lower_bound=1.5,
                upper_bound=2.5)
        param.getSample()
        param.evalPDF(2)
        self.assertEqual(param.evalPDF(2), myPDF(2))

    def testHM(self):

        theHM = HorsetailMatching(TP0, UncertainParameter('uniform'))
        theHM.evalMetric([0, 1])
        theHM = HorsetailMatching(TP0, UncertainParameter('interval'))
        theHM.evalMetric([0, 1])
        theHM = HorsetailMatching(TP0, UncertainParameter('gaussian'))
        theHM.evalMetric([0, 1])
        theHM = HorsetailMatching(TP0, [UncertainParameter('gaussian')])
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

        uparams = [UncertainParameter('uniform'),
                UncertainParameter('uniform')]
        theHM = HorsetailMatching(fqoi, uparams, ftarget=ftarget)
        theHM.evalMetric([0, 0], method='empirical')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        theHM.uncertain_parameters = uparams
        theHM.evalMetric([0, 0], method='kernel')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        uparams = [UncertainParameter('uniform'),
                UncertainParameter('interval')]
        theHM = HorsetailMatching(fqoi, uparams, ftarget=ftarget,
                samples_prob=5, samples_int=3)

        theHM.evalMetric([0,0], method='kernel')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        uparams = [UncertainParameter('uniform'),
                UncertainParameter('interval'),
                UncertainParameter('gaussian')]
        theHM.uncertain_parameters = uparams

        theHM.evalMetric([0,0], method='kernel')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()
        theHM.evalMetric([0,0], method='empirical')
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        theHM = HorsetailMatching(fqoi, uparams,
                verbose=True, reuse_samples=True)

        uparams = [UncertainParameter('interval'),
                UncertainParameter('interval')]
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

        uparams = [UncertainParameter('uniform'),
                UncertainParameter('interval')]
        theHM = HorsetailMatching(fboth, uparams, jac=True, samples_prob=5,
                samples_int=3, verbose=True, reuse_samples=True)

        theHM.evalMetric([1, 1])
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()
        with self.assertRaises(TypeError):
            theHM.evalMetric([0, 1], method='empirical')

        theHM.fqoi = fqoi
        theHM.jac = fgrad
        theHM.evalMetric([1, 1])
        (x1, y1), (x2, y2), CDFs = theHM.getHorsetail()

        theHM.uncertain_parameters = UncertainParameter('uniform')
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

        uparams = [UncertainParameter('uniform')]
        theHM = HorsetailMatching(fqoi, uparams, ftarget=ftarget,
                integration_points=np.linspace(0.99, 1.01, 100),
                kernel_bandwidth=0.0001)
        ans = theHM.evalMetric([0])
        self.assertAlmostEqual(ans, np.sqrt(2), places=5)

        ans = theHM.evalMetric([0], method='kernel')
        self.assertAlmostEqual(ans, np.sqrt(2), places=5)

        ftarget = lambda h: -h
        fqoi = lambda x, u: np.linalg.norm(u)

        up = [UncertainParameter('uniform'), UncertainParameter('interval')]
        theHM = HorsetailMatching(fqoi, up, ftarget=(ftarget, ftarget),
                samples_prob=100, samples_int=50)
        ans = theHM.evalMetric([0])
        self.assertTrue(abs(ans - 2.05) < 5e-2)

        ftarget = lambda h: -h
        fqoi = lambda x, u: u
        up = UncertainParameter('uniform')
        theHM = HorsetailMatching(fqoi, up, ftarget=ftarget,
                samples_prob=1000)
        ans = theHM.evalMetric([0])
        self.assertTrue(abs(ans - 1.4) < 1e-1)

        up = UncertainParameter('interval')
        theHM.samples_int=50
        theHM.uncertain_parameters = up
        print(theHM.evalMetric([1]))



if __name__ == "__main__":
    unittest.main()
