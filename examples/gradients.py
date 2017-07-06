from scipy.optimize import minimize
import matplotlib.pyplot as plt
import numpy as np

from horsetailmatching import UncertainParameter, HorsetailMatching
from horsetailmatching import UniformParameter, GaussianParameter
from horsetailmatching.demoproblems import TP1

def main():

    u1 = UniformParameter(lower_bound=-1, upper_bound=1)
    u2 = GaussianParameter(mean=0, standard_deviation=1)

    def fun_q(x, u):
        return TP1(x, u, jac=False)
    def fun_jac(x, u):
        return TP1(x, u, jac=True)[1]
    def fun_both(x, u):
        return TP1(x, u, jac=True)

    theHM = HorsetailMatching(fun_both, [u1, u2], jac=True, method='kernel',
            kernel_bandwidth=0.001, samples_prob=2000,
            integration_points=np.linspace(-1, 5, 500))

    theHM = HorsetailMatching(fun_q, [u1, u2], jac=fun_jac, method='kernel',
            kernel_bandwidth=0.001, samples_prob=2000,
            integration_points=np.linspace(-1, 5, 500))

    print(theHM.evalMetric(x=[1, 3]))

    solution = minimize(theHM.evalMetric, x0=[1, 3], method='BFGS', jac=True)
    print(solution)

    theHM.evalMetric(solution.x)
    (x, y), _, _ = theHM.getHorsetail()
    plt.plot(x, y, 'r', label='Optimum CDF')
    plt.plot([theHM.ftarget(yi) for yi in y], y, 'k--', label='Target')
    plt.xlim([-1, 15])
    plt.legend(loc='lower right')
    plt.show()

if __name__ == "__main__":
    main()
