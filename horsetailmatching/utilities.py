import numpy as np
import copy
import math

def normCDF(x):
    return (1.0 + math.erf(x / math.sqrt(2.))) / 2.0

def makeIter(x):
    if isinstance(x, basestring):
        return x
    try:
        iter(x)
        return [xi for xi in x]
    except:
        return [x]

def choose(n, k):
    if 0 <= k <= n:
        ntok,ktok = 1,1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0

def finDiff(fobj, dv, f0=None, eps=10**-6):
    return finiteDifference(fobj, dv, f0=None, eps=10**-6)

def finiteDifference(fobj, dv, f0=None, eps=10**-6):

    if f0 is None:
        f0 = fobj(dv)

    try:
        iter(dv)
    except:
        dv = [dv]

    if len(dv) == 1:
        fbase = copy.copy(f0)
        fnew = fobj(dv[0] + eps)
        return float((fnew - fbase)/eps)
    else:
        grad = []
        for ii in range(len(dv)):
            fbase = copy(f0)
            x = copy(dv)
            x[ii] += eps
            fnew = fobj(x)
            grad.append(float((fnew - fbase)/eps))
            return grad
