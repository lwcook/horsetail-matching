import matplotlib.pyplot as plt
from scipy.optimize import minimize

from horsetailmatching import UncertainParameter, HorsetailMatching
from horsetailmatching import UniformParameter, IntervalParameter
from horsetailmatching.demoproblems import TP2

def main():

    u1 = IntervalParameter(lower_bound=-1, upper_bound=1)
    u2 = UniformParameter(lower_bound=-1, upper_bound=1)

    def my_target(h):
        return 1

    theHM = HorsetailMatching(TP2, [u1, u2],
                              ftarget=(my_target, my_target),
                              samples_prob=500, samples_int=50)

    print(theHM.evalMetric([2,3]))

    upper, lower, CDFs = theHM.getHorsetail()
    for CDF in CDFs:
        plt.plot(CDF[0], CDF[1], 'grey', lw=0.5)
    plt.plot(upper[0], upper[1], 'b')
    plt.plot(lower[0], lower[1], 'b', label='Initial Horsetail Plot')

    solution = minimize(theHM.evalMetric, x0=[1,1], method='Nelder-Mead')
    print(solution)

    upper, lower, CDFs = theHM.getHorsetail()
    for CDF in CDFs:
        plt.plot(CDF[0], CDF[1], 'grey', lw=0.5)
    plt.plot(upper[0], upper[1], 'r')
    plt.plot(lower[0], lower[1], 'r', label='Optimum Horsetail Plot')
    plt.plot([theHM.ftarget[0](y) for y in lower[1]], lower[1], 'k--',
        label='Target')

    plt.xlim([-3, 15])
    plt.legend(loc='upper left')
    plt.show()

if __name__ == "__main__":
    main()
