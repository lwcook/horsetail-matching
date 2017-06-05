import matplotlib.pyplot as plt
from scipy.optimize import minimize

from horsetailmatching import UncertainParameter, HorsetailMatching
from horsetailmatching.demoproblems import TP2

u1 = UncertainParameter('interval', lower_bound=-1, upper_bound=1)
u2 = UncertainParameter('uniform', lower_bound=-1, upper_bound=1)

def my_target(h): return 1

theHM = HorsetailMatching(TP2, [u1, u2], 
                          ftarget_l=my_target, ftarget_u=my_target,
                          n_samples_prob=500, n_samples_int=50)

print(theHM.evalMetric([2,3]))

theHM.plotHorsetail('b')
plt.xlim([0, 15])
plt.show()

solution = minimize(theHM.evalMetric, x0=[1,1], method='Nelder-Mead')
print(solution)

theHM.plotHorsetail('r')
plt.xlim([0, 15])
plt.show()
