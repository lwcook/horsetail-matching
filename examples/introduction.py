from scipy.optimize import minimize

import matplotlib.pyplot as plt
from horsetailmatching import UncertainParameter, UniformParameter
from horsetailmatching import GaussianParameter, HorsetailMatching
from horsetailmatching.demoproblems import TP1

def main():

    u1 = UniformParameter(lower_bound=-1, upper_bound=1)
    u2 = GaussianParameter(mean=0, standard_deviation=1)

    def my_target(h): return 0

    theHM = HorsetailMatching(TP1, [u1, u2], ftarget=my_target)

    print(theHM.evalMetric(x=[1, 3]))

    (x, y), _, _ = theHM.getHorsetail()
    plt.plot(x, y, 'b', label='Initial CDF')

    solution = minimize(theHM.evalMetric, x0=[1, 3], method='Nelder-Mead')
    print(solution)

    (x, y), _, _ = theHM.getHorsetail()
    plt.plot(x, y, 'r', label='Optimum CDF')
    plt.plot([theHM.ftarget(yi) for yi in y], y, 'k--', label='Target')
    plt.xlim([-1, 15])
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()
