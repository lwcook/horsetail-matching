import numpy as np
from copy import copy

def makeIter(x):
    try:
        iter(x)
        return [xi for xi in x]
    except:
        return [x]


def finDiff(fobj, dv, f0=None, dvi=None, eps=10**-6):
    return finiteDifference(fobj, dv, f0=None, dvi=None, eps=10**-6)

def finiteDifference(fobj, dv, f0=None, dvi=None, eps=10**-6):

    try:
        iter(dv)
    except:
        dv = [dv]

    if f0 is None: f0 = fobj(dv)
    if dvi is None:
        grad = []
        for ii in range(len(dv)):
            fbase = copy(f0)
            x = copy(dv)
            x[ii] += eps
            fnew = fobj(x)
            grad.append(float((fnew - fbase)/eps))
        if len(grad) == 1:
            return float(grad[0])
        else:
            return grad
    else:
        x = copy.copy(dv)
        x[dvi] += eps
        return (fobj(x) - f0) / eps
