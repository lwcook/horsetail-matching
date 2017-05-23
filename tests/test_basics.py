import os
import sys
import unittest
import pdb
import numpy as np
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
'../horsetailmatching/')))

from hm import HorsetailMatching
from parameters import ProbabilisticParameter, IntervalParameter
from demoproblems import TP1

class TestInitializations(unittest.TestCase):


    def testProbabilisticParameters(self):

        param = ProbabilisticParameter()
        self.assertEqual(param.distribution, 'uniform')

        param = ProbabilisticParameter('uniform')
        self.assertEqual(param.distribution, 'uniform')

        with self.assertRaises(ValueError):
            param = ProbabilisticParameter('baddist')

        param = ProbabilisticParameter(distribution='uniform', lower_bound=-1,
                upper_bound=1)
        self.assertAlmostEqual(param.mu, 0.)
        self.assertAlmostEqual(param.std, 1./np.sqrt(3.))

        param = ProbabilisticParameter(distribution='gaussian', mean=0.,
                standard_deviation=3.)
        self.assertAlmostEqual(param.mu, 0.)
        self.assertAlmostEqual(param.std, 3)

        fpdf = lambda x: 1./2.
        param = ProbabilisticParameter(distribution='custom', pdf=fpdf)

        with self.assertRaises(ValueError):
            param = ProbabilisticParameter(distribution='custom')


    def testIntervalParameters(self):

        param = IntervalParameter()

        param = IntervalParameter(lower_bound=-2, upper_bound=2)

    def testHM(self):

        uparams = [ProbabilisticParameter('uniform')]

        fqoi = lambda x, u: x + u
        ftarget = lambda h: 0

        theHM = HorsetailMatching(uparams, fqoi, ftarget)

        ans = theHM.evalMetricEmpirical([0,0])

class TestDemoProblems(unittest.TestCase):

    def testTP1(self):

        ans = TP1([0,0], [0,0])





if __name__ == "__main__":
    unittest.main()
