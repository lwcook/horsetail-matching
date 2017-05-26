import os
import sys
import unittest
import pdb
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
'../horsetailmatching/')))

from hm import HorsetailMatching
from parameters import UncertainParameter
from demoproblems import TP1

class TestInitializations(unittest.TestCase):


    def testUncertainParameter(self):

        param = UncertainParameter()
        self.assertEqual(param.distribution, 'uniform')

        param = UncertainParameter('uniform')
        self.assertEqual(param.distribution, 'uniform')

        with self.assertRaises(ValueError):
            param = UncertainParameter('baddist')

        param = UncertainParameter(distribution='uniform', lower_bound=-1,
                upper_bound=1)
        self.assertAlmostEqual(param.mu, 0.)
        self.assertAlmostEqual(param.std, 1./np.sqrt(3.))

        param = UncertainParameter(distribution='gaussian', mean=0.,
                standard_deviation=3.)
        self.assertAlmostEqual(param.mu, 0.)
        self.assertAlmostEqual(param.std, 3)

        fpdf = lambda x: 1./2.
        param = UncertainParameter(distribution='custom', pdf=fpdf)
        with self.assertRaises(ValueError):
            param = UncertainParameter(distribution='custom')

        with self.assertRaises(ValueError):
            param = UncertainParameter(distribution='custom')

        param = UncertainParameter('interval', lower_bound=-2, upper_bound=2)


    def testHM(self):

        uparams = [UncertainParameter('uniform')]

        fqoi = lambda x, u: x + u
        ftarget = lambda h: 0

        theHM = HorsetailMatching(fqoi, uparams, ftarget)

        ans = theHM.evalMetricEmpirical([0,0])

class TestDemoProblems(unittest.TestCase):

    def testTP1(self):

        ans = TP1([0,0], [0,0])


class TestHorsetailMatching(unittest.TestCase):

    ftarget = lambda h: 0

    def testProbMetric1D(self):

        uparams = [UncertainParameter('uniform')]
        theHM = HorsetailMatching(TP1, uparams)
        theHM.evalMetric([1, 1])

    def testProbMetric2D(self):

        uparams = [UncertainParameter('uniform'),
                   UncertainParameter('gaussian')]
        theHM = HorsetailMatching(TP1, uparams)
        theHM.evalMetric([1, 1])



if __name__ == "__main__":
    unittest.main()
