from scipy.optimize import minimize

import matplotlib.pyplot as plt
from horsetailmatching import UncertainParameter, HorsetailMatching
from horsetailmatching.demoproblems import TP3

u1 = UncertainParameter('uniform', lower_bound=-1, upper_bound=1)

def my_target(h):
    return 0 - 50*h**6

def my_func(x, u):
    if x < 0:
        return abs(x)*10 + 10
    elif x > 10:
        return abs(x-10)*10 + 10
    else:
        return TP3(x, u)

theHM = HorsetailMatching(my_func, u1, ftarget=my_target)

solution = minimize(theHM.evalMetric, x0=5, method='Nelder-Mead')
print(solution)

theHM.plotHorsetail()
plt.xlim([-10, 10])
plt.show()
