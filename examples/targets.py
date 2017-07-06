import numpy as np
import matplotlib.pyplot as plt
import pdb
from scipy.optimize import minimize

from horsetailmatching import HorsetailMatching, UncertainParameter
from horsetailmatching import UniformParameter, GaussianParameter
from horsetailmatching.demoproblems import TP3

def main():

    def plotHorsetail(theHM, c='b'):
        (x1, y1), (x2, y2), _ = theHM.getHorsetail()
        plt.plot(x1, y1, c=c)
        plt.plot([theHM.ftarget(y) for y in y1], y1, c=c, linestyle='dashed')

    u1 = GaussianParameter()

    def myFunc(x, u):
        return TP3(x, u, jac=False)

    def standardTarget(h):
        return 0
    theHM = HorsetailMatching(myFunc, u1, ftarget=standardTarget,
            samples_prob=2000)

    solution = minimize(theHM.evalMetric, x0=[0.6], method='COBYLA',
            constraints=[{'type': 'ineq', 'fun': lambda x: x},
                         {'type': 'ineq', 'fun': lambda x: 1-x}])
    print(solution)
    theHM.evalMetric(solution.x)
    plotHorsetail(theHM, c='b')

    def riskAverseTarget1(h):
        return 0 - 3*h**3
    theHM.ftarget = riskAverseTarget1
    solution = minimize(theHM.evalMetric, x0=[0.6], method='COBYLA',
            constraints=[{'type': 'ineq', 'fun': lambda x: x},
                         {'type': 'ineq', 'fun': lambda x: 1-x}])
    print(solution)
    theHM.evalMetric(solution.x)
    plotHorsetail(theHM, c='g')

    def riskAverseTarget2(h):
        return 0 - 10*h**10
    theHM.ftarget = riskAverseTarget2
    solution = minimize(theHM.evalMetric, x0=[0.6], method='COBYLA',
            constraints=[{'type': 'ineq', 'fun': lambda x: x},
                         {'type': 'ineq', 'fun': lambda x: 1-x}])
    print(solution)
    theHM.evalMetric(solution.x)
    plotHorsetail(theHM, c='r')

    plt.ylim([0, 1])
    plt.xlim([-10, 5])
    plt.plot([], [], 'k', label='CDF')
    plt.plot([], [], 'k--', label='Target')
    plt.legend(loc='lower left')
    plt.show()

if __name__ == "__main__":
    main()
