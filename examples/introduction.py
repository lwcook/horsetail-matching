from scipy.optimize import minimize

import matplotlib.pyplot as plt
from horsetailmatching import UncertainParameter, HorsetailMatching
from horsetailmatching.demoproblems import TP1

u1 = UncertainParameter('uniform', lower_bound=-1, upper_bound=1)
u2 = UncertainParameter('gaussian', mean=0, standard_deviation=1)

def my_target(h): return 0

theHM = HorsetailMatching(my_func, [u1, u2], ftarget=my_target)

print(theHM.evalMetric(x=[1,1]))

theHM.plotHorsetail('b')
plt.xlim([-0.1, 1])
plt.show()

solution = minimize(theHM.evalMetric, x0=[3,2], method='Nelder-Mead')
print(solution)
