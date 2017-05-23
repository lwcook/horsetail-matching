import numpy as np

def makeIter(x):
    try:
        iter(x)
        return [xi for xi in x]
    except:
        return [x]
