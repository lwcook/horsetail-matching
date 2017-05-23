import numpy as np

def TP1(dv, u):
    return np.linalg.norm(np.array(dv)) + np.linalg.norm(np.array(u))
