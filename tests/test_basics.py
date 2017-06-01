import os
import sys
import unittest
import pdb

import numpy as np
import matplotlib.pyplot as plt

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

        param = UncertainParameter('interval', lower_bound=-2, upper_bound=2)
        self.assertAlmostEqual(param.lb, -2)
        self.assertAlmostEqual(param.ub, 2)

    def testHM(self):


        fqoi = lambda x, u: np.linalg.norm(x) + np.linalg.norm(u)
        ftarget = lambda h: 0

        uparams = [UncertainParameter('uniform')]
        theHM = HorsetailMatching(fqoi, uparams, ftarg=ftarget)
        theHM.evalMetric([0,0], method='empirical')

        uparams = UncertainParameter('uniform')
        theHM = HorsetailMatching(fqoi, uparams, ftarg=ftarget)
        theHM.evalMetric([0,0], method='empirical')

        uparams = [UncertainParameter('uniform'),
                UncertainParameter('interval')]
        theHM = HorsetailMatching(fqoi, uparams, ftarg=ftarget)
        theHM.evalMetric([0,0], method='kernel')


class TestHorsetailMatching(unittest.TestCase):


    def testMetricValues(self):

        ftarget = lambda h: 0
        fqoi = lambda x, u: 1

        uparams = [UncertainParameter('uniform')]
        theHM = HorsetailMatching(fqoi, uparams, ftarg=ftarget,
                q_integration_points=np.linspace(0.99, 1.01, 100), bw=0.0001)
        ans = theHM.evalMetric([0])
        self.assertAlmostEqual(ans, np.sqrt(2), places=5)

        ans = theHM.evalMetric([0], method='kernel')
        self.assertAlmostEqual(ans, np.sqrt(2), places=5)

        ftarget = lambda h: -h
        fqoi = lambda x, u: np.linalg.norm(u)

        up = [UncertainParameter('uniform'), UncertainParameter('interval')]
        theHM = HorsetailMatching(fqoi, up, ftarg_u=ftarget, ftarg_l=ftarget,
                n_samples_prob=100, n_samples_int=50)
        ans = theHM.evalMetric([0])
        self.assertTrue(abs(ans - 2.05) < 5e-2)

        ftarget = lambda h: -h
        fqoi = lambda x, u: u
        up = UncertainParameter('uniform')
        theHM = HorsetailMatching(fqoi, up, ftarg=ftarget,
                n_samples_prob=1000)
        ans = theHM.evalMetric([0])
        self.assertTrue(abs(ans - 1.4) < 1e-1)


if __name__ == "__main__":
    unittest.main()
