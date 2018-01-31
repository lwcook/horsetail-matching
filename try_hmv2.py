import os
import sys
import unittest
import pdb
import copy
import warnings

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
    'horsetailmatching/')))

from hm import HorsetailMatching
from densitymatching import DensityMatching
from weightedsum import WeightedSum
from parameters import GaussianParameter, UniformParameter, IntervalParameter
from parameters import UncertainParameter
from surrogates import PolySurrogate
from demoproblems import TP0, TP1, TP2, TP3, TP2b


def main():

    theHM = HorsetailMatching(TP0, GaussianParameter(),
            integration_points=np.linspace(-1, 100, 100))

    theHM.evalMetric([0, 1])

    theHM = HorsetailMatching(TP0, UniformParameter(), IntervalParameter())
    theHM.evalMetric([0, 1])
    theHM = HorsetailMatching(TP0, UniformParameter(), [IntervalParameter()])
    theHM.evalMetric([0, 1])
    theHM = HorsetailMatching(TP0, [GaussianParameter()])
    theHM.evalMetric([0, 1])


    def fqoi(x, u):
        return TP1(x, u, jac=False)

    def fgrad(x, u):
        return TP1(x, u, jac=True)[1]

    def fboth(x, u):
        return TP1(x, u, jac=True)

    def ftarget(h):
        return 0


    theDM = DensityMatching(TP0, GaussianParameter())
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

#    _ = theDM.uncertain_parameters

#    _ = theDM.ftarget

    theDM.u_samples = None

if __name__ == "__main__":
    main()
