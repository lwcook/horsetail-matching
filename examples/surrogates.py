import numpy as np

from horsetailmatching import HorsetailMatching, IntervalParameter
from horsetailmatching import UniformParameter, GaussianParameter
from horsetailmatching.demoproblems import TP2
from horsetailmatching.surrogates import PolySurrogate

from pyKriging.krige import kriging
from pyKriging.samplingplan import samplingplan
from equadratures import Polyreg

def main():

    uparams = [UniformParameter(), IntervalParameter()]
    theHM = HorsetailMatching(TP2, uparams)
    print('Metric evaluated with direct sampling: ', theHM.evalMetric([0, 1]))

    thePoly = PolySurrogate(dimensions=len(uparams), order=3)
    u_quadrature = thePoly.getQuadraturePoints()

    def nipcSurrogate(u_quad, q_quad):
        thePoly.train(q_quad)
        return thePoly.predict

    theHM.surrogate = nipcSurrogate
    theHM.surrogate_points = u_quadrature
    print('Metric evaluated with polynomial chaos surrogate: ', theHM.evalMetric([0, 1]))

    sp = samplingplan(2)
    u_sampling = sp.optimallhc(25)

    def krigSurrogate(u_lhc, q_lhc):
        krig = kriging(u_lhc, q_lhc)
        krig.train()
        return krig.predict

    theHM.surrogate = krigSurrogate
    theHM.u_quadrature_points = u_sampling
    print('Metric evaluated with kriging surrogate: ', theHM.evalMetric([0, 1]))

    U1, U2 = np.meshgrid(np.linspace(-1, 1, 5), np.linspace(-1, 1, 5))
    u_tensor = np.vstack([U1.flatten(), U2.flatten()]).T

    def pSurrogate(u_tensor, q_tensor):
        poly = Polyreg(np.mat(u_tensor), np.mat(q_tensor).T, 'quadratic')
        def model(u):
            return poly.testPolynomial(np.mat(u))
        return model

    theHM.surrogate = pSurrogate
    theHM.u_quadrature_points = u_tensor
    print('Metric evaluated with quadratic surrogate: ', theHM.evalMetric([0, 1]))

if __name__ == "__main__":
    main()
