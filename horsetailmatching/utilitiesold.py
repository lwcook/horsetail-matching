import copy
import pdb
import numpy as np
import math
import scipy
import scipy.special as scp
import os
import subprocess
import time

from scipy.stats import beta

import pickle
import json

import datetime

import matplotlib as mpl
import matplotlib.pyplot as plt

# TO USE THESE FUNCTIONS, ADD THE FOLLOWING TO YOUR PYTHON FILE:
# import sys
# home_dir = os.getenv("HOME")
# sys.path.insert(0, home_dir + '/phd-code/utils/')
# import utilities as utils


def test_const(x, u):
    return 0.0


def test_constr(x, u):
    return 0.0


def TP1x1u(x, u):

    try:
        iter(u)
        u = u[0]
    except:
        pass

    try:
        iter(x)
        x = x[0]
    except:
        pass

    c = math.atan(x + 0.3)
    return 1. + 8.*c + (1./c)*(np.exp(1.5*u) - 1)

def TP3Dopt(x, u, delta=1):

    try:
        iter(u)
        u = u[0]
    except:
        pass

    return x[0]**2 + 2*u*x[1] + x[2]*u**2 + 5
#    return x[0]**2 + 2*u*x[1] + 1*x[2]*abs(u)**2 + 5 + 0.1*x[2]


def TP2xNu(x, u, bExp=True):
    y = x[0]/2
    z = x[1]/2

    qu = 0.0
    if bExp:  # Exponential Test Function

        qrosen = ((2-y)**2 + 2*(z - y**2)**2)/5
        for ii, ui in enumerate(u):
            scale = (1 + 0.1*z**2 + 0.2*y)
            qu += (1./(ii+1.)) * scale*ui

        qu = np.exp(-qu)
        return 1 + qu + qrosen

    else:  # Reciprocal Test Function
        for ii, ui in enumerate(u):
            scale = (1 + 0.1*y + 0.2*z)/5.
#            if ii % 2 == 0:
#                scale = (1 + 0.1*z**2 + 0.2*y)
#            else:
#                scale = (1 + 0.1*y**2 + 0.2*z)
            qu += scale*ui

        qu = 1./(1. + qu)

        return qu - 1.5


def TP4x2u(x, u, delta=1):
    '''x should be 4D, u should be 2D'''

    return x[0]**2 + 2*u[0]*(x[1] + x[3]) + (x[2] + x[3])*u[1]**2 + 5


def TP2Dopt(x, u, delta=1):
    try:
        iter(u)
        u = u[0]
    except:
        u = u

    return (x[0] + 0.5*x[1])**2*u + x[1]*u**2 + 5


def TP2x2uOld(x, u):
    '''Both x and u should be 2D'''
    return x[0]**2 + 2*u[0]*x[0] + x[1]*u[1]**2 + 5


def TP2D(x, u):
    '''Both x and u should be 2D'''
    y = x[0]/5
    z = x[1]/5
    return 1 + ((-1-y)**2 + 10*(z - y**2)**2)/2 + \
        z*math.sinh(u[0])*math.sinh(u[1]) - y*math.cosh(u[1])


def TP2x2u(x, u):
    '''Both x and u should be 2D'''

#    return (y0)**2 + (y1)*(v1**2)**.9 + y2*(v2**2)**.7 + 10
#    return y0 + (y1)*5*u[0]**2*v2 + y2*u[1]**2 + 10
#    return x[0]**2 + 2*u[0]*x[0] + x[1]*u[1]**2 + 5
#    return 10 + 0.33*(y1**2 + y2**2)**0.5 + \
#        (1 + (y1))*2*u[0]*(u[1]**2) + \
#        (y2/(y1**2+1))*0.5*u[1]
    y = x[0]/2.
    z = x[1]/2. + 12
    q =  0.25*((y**2+z**2)/40. + 5*u[0]*u[1]*y - z*u[1]**2) \
        + 0.2*(z*u[1]**3) + 10

    return q

def TP2xIS(x, u):
    y = x[0]/2
    z = x[1]/2
    qrosen = ((2-y)**2 + 2*(z - y**2)**2)/5

#    qu = (1./1.) + (1 + 0.1*z**2)*u[0] + (1./2.) + (1 + 0.1*y**2)*u[1]
#    qu = np.exp(-qu)*2

#    qu = 0.5*(np.linalg.norm(u)*3 + 0.02*u[0]**3)
    qu = (u[0]+1)**2

    try:
        r = u[0]
    except:
        r = u
#    if r < 0.33:
#        qu += 1

    return 1 + (qrosen + qu)

def TP10x1u(x, u):
    '''Both x and u should be 2D'''
    try:
        iter(u)
        u = u[0]
    except:
        u = u

    qrosen = ((2-u)**2 + 2*(u - u**2)**2)/5

    qx = 0.0

    for ii, xi in enumerate(x):
        scale = (1 + 0.2*u**2)
        qx += (1./(ii+1.)) * scale*xi

    qx = np.exp(-qx)

    return 1 + (qrosen + qx)

def TP1x1u(x, u):
    '''Both x and u should be 2D'''
    try:
        iter(u)
        u = u[0]
    except:
        u = u

    qrosen = ((2-u)**2 + 2*(u - u**2)**2)/5

    qx = 0.0

    scale = (1 + 0.2*u**2)
    qx += (1./(ii+1.)) * scale*x

    qx = np.exp(-qx)

    return 1 + (qrosen + qx)


def TP2x2uGrad(x, u):
    '''Both x and u should be 2D'''

#    return (y0)**2 + (y1)*(v1**2)**.9 + y2*(v2**2)**.7 + 10
#    return y0 + (y1)*5*u[0]**2*v2 + y2*u[1]**2 + 10
#    return x[0]**2 + 2*u[0]*x[0] + x[1]*u[1]**2 + 5
#    return 10 + 0.33*(y1**2 + y2**2)**0.5 + \
#        (1 + (y1))*2*u[0]*(u[1]**2) + \
#        (y2/(y1**2+1))*0.5*u[1]
    y = x[0]/2.
    z = x[1]/2. + 12
    q =  0.25*((y**2+z**2)/40. + 5*u[0]*u[1]*y - z*u[1]**2) \
        + 0.2*(z*u[1]**3) + 10

    dqdx1 = (1./8.)*( (2*y)/40. + 5*u[0]*u[1])
    dqdx2 = (1./8.)*( (2*z)/40. - u[1]**2) + 0.1*u[1]**3

    return q, [dqdx1, dqdx2]


def IRTP_2x1u(x, u):

    try:
        iter(u)
        u = u[0]
    except:
        u = u

#    return 1 + 8 * np.arctan(2*x[0]+x[1]+0.3) + (1./np.arctan(x[0]+2*x[1]+0.3)) *\
#            (np.exp(1.5*u) - 1)

    return x[0] + (1 + 0.2*np.linalg.norm(x))*abs(u)


def IRTP_g(x, u):

    try:
        iter(u)
        u = u[0]
    except:
        u = u

    return -(x[1] + (abs(x[0]) + 1)*u/2.) + 2


def finite_diff(fobj, dv, f0=None, dvi=None, eps=10**-6):

    try:
        iter(dv)
    except:
        dv = [dv]

    if f0 is None: f0 = fobj(dv)
    if dvi is None:
        grad = []
        for ii in range(len(dv)):
            fbase = copy.copy(f0)
            x = copy.copy(dv)
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


def float_gen(iterable):
    for obj in iterable:
        try:
            float(obj)
            yield float(obj)
        except ValueError:
            pass

def savelog(log_name='saved_log', front_name='saved_front',
            bSaveBase=True, base='/phd-thesis/Figs/'):

    date = datetime.datetime.now()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    final_fname = 'RO_front_' + str(date.day) + months[date.month-1] + '.txt'
    final_lname = 'RO_log_' + str(date.day) + months[date.month-1] + '.txt'

    home = os.getenv("HOME")

    subprocess.call(["cp", log_name,
                     home + '/phd-code/RansAerofoil/results/' + final_lname])
    subprocess.call(["cp", front_name,
                     home + '/phd-code/RansAerofoil/results/' + final_fname])
    subprocess.call(["cp", log_name,
                     home + base + final_lname])
    subprocess.call(["cp", front_name,
                     home + base + final_fname])


def readlog(log_name='saved_log', front_name='saved_front', loc="results"):

    home = os.getenv("HOME")
    if loc.lower() == 'base':
        fileloc = home + '/phd-thesis/Figs/'
    elif loc.lower() == 'results':
        fileloc = home + '/phd-code/RansAerofoil/results/'
    else:
        fileloc = os.getcwd()

    points = []
    with open(fileloc + log_name) as f:
       for line in f:
           floats = [_ for _ in float_gen(line.split())]
           if floats:
               points.append([floats[0], floats[1]])
    points = np.array(points)
    return points


def readfront(front_name='saved_front', loc='results'):

    home = os.getenv("HOME")
    if loc.lower() == 'base':
        fileloc = home + '/phd-thesis/Figs/'
    elif loc.lower() == 'results':
        fileloc = home + '/phd-code/RansAerofoil/results/'
    else:
        fileloc = os.getcwd()

    front = []
    with open(fileloc + front_name) as f:
       for line in f:
           floats = [_ for _ in float_gen(line.split())]
           if floats:
               front.append([floats[0], floats[1]])
    front = np.array(front)

    return front


def savefig(name='saved_fig', bSaveBase=False,
            base='/phd-thesis/Figs/', bSaveData=False, formatstr='pdf'):
    '''Function that saves the plot as well as the
    underlying data of the currently open figure:
    -name: string that the figure is saved as'''

    date = datetime.datetime.now()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    subprocess.call(["mkdir", "-p", "./figs/"])
#    plt.savefig('./output/' + str(name) + '.pdf', format='pdf')
    plt.savefig('./figs/' +  str(name) + '_' + str(date.day) +
                months[date.month-1] + '.' + formatstr, format=formatstr)

    if bSaveData:
        try:
            # Get dictionary of data in plot ready to save
            data = {}
            data['xlabel'] = plt.gca().get_xlabel()
            data['ylabel'] = plt.gca().get_ylabel()
            data['title'] = plt.gca().get_title()
            for line in plt.gca().get_lines():
                data[line.get_label()] = \
                    {'color': line.get_color(),
                     'linestyle': line.get_linestyle(),
                     'xdata': list(line.get_xdata()),
                     'ydata': list(line.get_ydata())}
            savedata(data, name=name, base=base, bSaveBase=bSaveBase)

        except:
            print('WARNING: Unable to save plot data in json format.')


def savedata(data, name='saved_data',
             bSaveBase=True, base='/phd-thesis/Figs/'):
    '''Save data in json format, passed in as a dictionary'''

    date = datetime.datetime.now()

    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    subprocess.call(["mkdir", "-p", "./results/"])

    home = os.getenv("HOME")
    with open('./results/' + str(name) + '_json.txt', 'w') as f:
        json.dump(data, f, indent=4)
    with open(home + base + str(name) + '_' + str(date.day) +
              months[date.month-1] + '_json.txt', 'w') as f:
                json.dump(data, f, indent=4)


def readdata(name='saved_data', bBase=False, base='/phd-thesis/Figs/'):
    if not bBase:
        with open('./results/' + str(name) + '_json.txt', 'r') as f:
            jdict = json.load(f)
    else:
        home = os.getenv("HOME")
        with open(home + base + str(name) + '_json.txt', 'r') as f:
            jdict = json.load(f)
    return jdict


def replot(name='saved_data', bBase=False,
        base='/phd-thesis/Figs/', lstyle=None):

    jdict = readdata(name, bBase, base)

    for key in jdict.keys():
        if key != 'xlabel' and key != 'ylabel' and key != 'title':
            if jdict[key]['xdata']:
                if lstyle is None:
                    style = jdict[key].setdefault('linestyle', '-')
                else:
                    style = lstyle
                plt.plot(jdict[key]['xdata'], jdict[key]['ydata'],
                         c=jdict[key].setdefault('color', 'k'),
                         linestyle=style,
                         label=jdict[key].setdefault('label', []))
        elif key == 'xlabel':
            plt.xlabel(jdict[key])
        elif key == 'ylabel':
            plt.ylabel(jdict[key])


def readplot(name='saved_data', bBase=False, base='/phd-thesis/Figs'):

    jdict = readdata(name, bBase, base)

    xdatas, ydatas, labels, linestyles, colors = [], [], [], [], []
    info = {}
    for key in jdict.keys():

        if key != 'xlabel' and key != 'ylabel' and key != 'title':

            if jdict[key]['xdata']:  # Check the line contains data
                xdatas.append(jdict[key]['xdata'])
                ydatas.append(jdict[key]['ydata'])
                labels.append(jdict[key])
                linestyles.append(jdict[key]['linestyle'])
                colors.append(jdict[key]['color'])

        elif key == 'xlabel':
            info['xlabel'] = jdict[key]
        elif key == 'ylabel':
            info['ylabel'] = jdict[key]
        elif key == 'title':
            info['title'] = jdict[key]

    return xdatas, ydatas, info


def mpl2tex(figsize=[2.8, 2.2]):
        '''Makes matplotlib's parameters give a plot formatted
        for latex documents
        -figsize: the figures dimension's in inches.
        Default is small enough for single column'''

        # Note: 1.0/72.27 inches per pt
        # [2.8,2.2] fits on the smallest of these (CMAME) and is a good ratio
        # CMAME template has a 390.0 pt wide textwidth - 5.396 inches
        # Thesis: 437.46 - 6.05 inches

        mpl.rcParams.update({"figure.figsize": figsize,
                             "font.family": "serif",
                             "text.usetex": True,
                             "text.latex.preamble": r"\usepackage{amsmath}",
                             "font.size": 8,
                             "font.weight": "light",
                             'axes.labelsize': 9,
                             'axes.titlesize': 8,
                             'legend.fontsize': 8,
                             'xtick.labelsize': 8,
                             'ytick.labelsize': 8,
                             'lines.linewidth': 0.6,
                             'axes.linewidth': 0.75,
                             'patch.linewidth': 0.75,
                             'legend.fontsize': 'medium',
                             'legend.scatterpoints': 1
                             })



def array_from_log(filename='RO_log.txt', num_objs=2):

    pointslist, candslist = [], []
    with open(filename, 'r') as f:
        for il, line in enumerate(f):
            if line.split():
                floats = [val for val in float_gen(line.split())]
                pointslist.append(floats[0:num_objs])
                candslist.append(floats[num_objs:])

    points = np.zeros([len(pointslist), len(pointslist[0])])
    for ir, row in enumerate(pointslist):
        points[ir, :] = row

    cands = np.zeros([len(candslist), len(candslist[0])])
    for ir, row in enumerate(candslist):
        cands[ir, :] = row

    return points, cands


def is_dominated_by(point, other, num_objs=2):
    '''Check whether point is dominated by other:
    i.e. if every entry in other is less than or
    equal to those in point'''
    if point == other:
        return False
    else:
        bDominated = True
        for ii in range(num_objs):
            if point[ii] < other[ii]:
                bDominated = False
        return bDominated


def pareto_from_log(filename='RO_log.txt', num_objs=2):
    ''' Gets pareto front from a RO log file in my own format
    of Objs: ... Cands: ...
    '''

    points, cands = array_from_log(filename, num_objs)

    return pareto_from_points(points, cands)


def pareto_from_points(objs, designs):
    ''' Gets pareto front from a RO log file in my own format
    of Objs: ... Cands: ...
    '''
    points = np.array(objs)
    cands = np.array(designs)
    num_objs = len(points[0])

    pfront, pset = [], []
    for irow in range(points.shape[0]):
        point = list(points[irow, :])
        cand = list(cands[irow, :])
        if pfront:
            toRemove, toRemoveCand, bToAdd = [], [], True
            for io, other in enumerate(pfront):
                if point == other:
                    bToAdd = False
                    break
                elif is_dominated_by(point, other, num_objs):
                    bToAdd = False
                elif is_dominated_by(other, point, num_objs):
                    toRemove.append(other)
                    toRemoveCand.append(pset[io])

            for rempoint, remcand in zip(toRemove, toRemoveCand):
                pfront.remove(rempoint)
                pset.remove(remcand)

            if bToAdd:
                pfront.append(point)
                pset.append(cand)

        else:
            pfront.append(point)
            pset.append(cand)

    pfrontarr = np.zeros([len(pfront), len(pfront[0])])
    psetarr = np.zeros([len(pset), len(pset[0])])
    for ir in range(len(pfront)):
        pfrontarr[ir, :] = pfront[ir]
        psetarr[ir, :] = pset[ir]

    sortedi = np.argsort(pfrontarr, axis=0)[:, 0]

    pfrontsorted = pfrontarr[sortedi, :]
    psetsorted = psetarr[sortedi, :]

    return pfrontsorted, psetsorted


def plotcontours(f, lb, ub, N=50):
    xlin = np.linspace(lb[0], ub[0], N)
    ylin = np.linspace(lb[1], ub[1], N)

    xv, yv, zv = np.zeros([N, N]), np.zeros([N, N]), np.zeros([N, N])
    for ix, x in enumerate(xlin):
        for iy, y in enumerate(ylin):
            xv[iy, ix] = float(x)
            yv[iy, ix] = float(y)
            zv[iy, ix] = f([float(x), float(y)])

    return xv, yv, zv


def savecontours(xv, yv, zv, N, name='saved_contours',
                 bSaveBase=True, base='/phd-thesis/Figs/'):
    jdict = {'xv': [_ for _ in xv.flatten()],
             'yv': [_ for _ in yv.flatten()],
             'zv': [_ for _ in zv.flatten()],
             'N': N}
    savedata(jdict, name=name)


def readcontours(name='saved_contours'):
    jdict = readdata(name=name)
    N = jdict['N']
    xv = np.array(jdict['xv']).reshape([N, N])
    yv = np.array(jdict['yv']).reshape([N, N])
    zv = np.array(jdict['zv']).reshape([N, N])
    return xv, yv, zv, N


def step(x):
    return 1 * (x > 0)


def ramp(x, width):
    return minsmooth(1, maxsmooth(0, (x - width/2)*(1/width)))


def trint(x, width):
    w = width/2.
    xb = maxsmooth(-w, minsmooth(x, w))
    y1 = 0.5 + xb/w + xb**2/(2*w**2)
    y2 = xb/w - xb**2/(2*w**2)
    return minsmooth(y1, 0.5) + maxsmooth(y2, 0.0)


def minsmooth(a, b, eps=0.0000):
    return 0.5*(a + b - np.sqrt((a-b)**2 + eps**2))


def maxsmooth(a, b, eps=0.0000):
    return 0.5*(a + b + np.sqrt((a-b)**2 + eps**2))


def extalg(xarr, alpha=10):
    '''Given an array xarr of values, smoothly return the max/min'''
    return sum(xarr * np.exp(alpha*xarr))/sum(np.exp(alpha*xarr))


def extgrad(xarr, alpha=10):
    '''Given an array xarr of values, return the gradient of the smooth min/max
    swith respect to each entry in the array'''
    term1 = np.exp(alpha*xarr)/sum(np.exp(alpha*xarr))
    term2 = 1 + alpha*(xarr - extalg(xarr, alpha))

    return term1*term2


def rosenbrock(x):
    return (x[1]-x[0]**2)**2 + (1-x[0])**2


def lf_rosenbrock(x):
    return x[0]**4 + x[1]**2


def choose(n, k):
    if 0 <= k <= n:
        ntok, ktok = 1, 1
        for t in xrange(1, min(k, n - k) + 1):
            ntok *= n
            ktok *= t
            n -= 1
        return ntok // ktok
    else:
        return 0


def LHS(sample_num, dims=1):
    # Basic LHS that samples over the hypercube [-1,1]

    sample_points = np.zeros([sample_num, dims])
    permutations = np.zeros([sample_num, dims], int)

    # Using a uniform distribution
    for idim in range(dims):

        segment_size = 2. / float(sample_num)

        for isample in range(0, sample_num):

            segment_min = -1 + isample*segment_size
            sample_points[isample, idim] = segment_min + \
                np.random.uniform(0, segment_size)

        permutations[:, idim] = np.random.permutation(sample_num)

    temp = sample_points*0
    for isample in range(0, sample_num):
        for idim in range(0, dims):
            temp[isample, idim] = \
                sample_points[permutations[isample, idim], idim]
    sample_points = temp

    return sample_points

def gridsample(sample_num, dims=1):

    if dims == 1:
        return [[_] for _ in np.linspace(-1, 1, sample_num)]
    elif dims == 2:
        vx = np.linspace(-1, 1, int(np.sqrt(sample_num)))
        vy = np.linspace(-1, 1, int(np.sqrt(sample_num)))
        mx, my = np.meshgrid(vx, vy)

        return [[u[0], u[1]] for u in zip(mx.flatten(), my.flatten())]

def randomsample(sample_num, dims=1):
    sample_points = []
    for isample in range(sample_num):
        point = []
        for idim in range(dims):
            point.append(np.random.random())
        sample_points.append(point)
    return sample_points

def kernel(points, M=None, bw=None, ktype='gauss', bGrad=False):

    if M is None:
        M = np.array(points).size
    if bw is None:
        bw = (4./(3.*M))**(1./5.)*np.std(points)

    # NB make evaluations matrix compatible
    if ktype == 'gauss' or ktype == 'gaussian':
        KernelMat = (1./M)*scp.ndtr(points/bw)
    elif ktype == 'gemp':
        bwemp = bw/100.
        KernelMat = (1./M)*scp.ndtr(points/bwemp)
    elif ktype == 'step' or ktype == 'empirical':
        KernelMat = (1./M)*step(points)
    elif ktype == 'uniform' or ktype == 'uni':
        KernelMat = (1./M)*ramp(points, width=bw*np.sqrt(12))
    elif ktype == 'triangle' or ktype == 'tri':
        KernelMat = (1./M)*trint(points, width=bw*2.*np.sqrt(6))

    if bGrad:
        if ktype == 'gauss' or ktype == 'gaussian':
            const_term = 1.0/(M * np.sqrt(2*np.pi*bw**2))
            KernelGradMat = const_term * np.exp(-(1./2.) * (points/bw)**2)
        elif ktype == 'gemp':
            const = 1.0/(M * np.sqrt(2*np.pi*bwemp**2))
            KernelGradMat = const * np.exp(-(1./2.) * (points/bwemp)**2)
        elif ktype == 'uniform' or ktype == 'uni':
            width = bw*np.sqrt(12)
            const = (1./M)*(1./width)
            KernelGradMat = const*(step(points+width/2) -
                                   step(points-width/2))
        elif ktype == 'triangle' or ktype == 'tri':
            width = bw*2.*np.sqrt(6)
            const = (1./M)*(2./width)
            KernelGradMat = const*(ramp(points+width/4, width/2) -
                                   ramp(points-width/4, width/2))
        else:
            KernelGradMat = 0*points
            print('Warning: kernel type gradient not supported')

        return KernelMat, KernelGradMat
    else:
        return KernelMat


def eval_quad_points(u_sparse, dv, f_obj, nu,
                     f_grad=False, bGrad=False, eps=10**-6):
    '''Evaluate the actual simulation of the quantity of interest at a
    sparse grid of quadrature points over the uncertainty space at a given
    vector of design variables
        - u_sparse: quadrature points
        - dv: design variables
        - f_obj: simulation of quantity of interest
        - f_grad: simulation of gradient of qoi
        - bGrad: whether to evaluate the gradient
        - eps: finite differencing tolerance '''

    nu_tot = nu
    f_sparse = np.zeros(u_sparse.size / nu_tot)
    ndv = len(dv)
    if bGrad:
        grad_sparse = np.zeros([u_sparse.size / nu_tot, ndv])
        for ii, ui in enumerate(u_sparse):
            if callable(f_grad):
                f_sparse[ii] = f_obj(dv, ui)
                grad_sparse[ii, :] = f_grad(dv, ui)
            elif f_grad:
                f_sparse[ii], grad_sparse[ii, :] = f_obj(dv, ui)
            else:
                f_sparse[ii] = f_obj(dv, ui)
                grad_sparse[ii, :] = finite_diff(
                    lambda dv: f_obj(dv, ui), dv, f0=f_sparse[ii], eps=eps)

        return f_sparse, grad_sparse
    else:
        for ii, ui in enumerate(u_sparse):
            f_sparse[ii] = f_obj(dv, ui)
        return f_sparse


def surrogate(x_sp, f_sp, x_den, dims=1):
    """ Creates a simpler polynomial surrogate to
    model a function over the range [-1,1]**dims """

    if dims == 1:
        X, F = x_sp.flatten(), f_sp.flatten()

        A = np.array([X * 0. + 1., X, X**2, X**3, X**4, X**5]).T
        c, r, rank, s = np.linalg.lstsq(A, F)

        def poly_model(u):
            return c[0] + c[1]*u + c[2]*u**2 + c[3]*u**3 +\
                c[4]*u**4 + c[5]*u**5

        return poly_model(x_den)

    elif dims == 2:

        X, Y = x_sp[:, 0].flatten(), x_sp[:, 1].flatten()
        F = f_sp.flatten()

        A = np.array(
            [X*0+1., X, Y, X**2, X*Y, Y**2, X**3, X**2*Y, X*Y**2, Y**3]).T
        c, r, rank, s = np.linalg.lstsq(A, F)

        def poly_model(u):
            x = u[:, 0]
            y = u[:, 1]
            return c[0] + c[1]*x + c[2]*y + \
                c[3]*x**2 + c[4]*x*y + c[5]*y**2 + \
                c[6]*x**3 + c[7]*x**2*y + c[8]*x*y**2 + c[9]*y**3

        return poly_model(x_den)

    elif dims == 3:

        X = x_sp[:, 0].flatten()
        Y = x_sp[:, 1].flatten()
        Z = x_sp[:, 2].flatten()
        F = f_sp.flatten()

        A = np.array(
            [X*0+1.,
             X, Y, Z,
             X**2, Y**2, Z**2,
             X*Y, X*Z, Y*Z,
             X**3, Y**3, Z**3,
             X**2*Y, X**2*Z, Y**2*X,
             Y**2*Z, Z**2*X, Z**2*Y,
             X*Y*Z
            ]).T

        c, r, rank, s = np.linalg.lstsq(A, F)

        def poly_model(u):
            x = u[:, 0]
            y = u[:, 1]
            z = u[:, 2]
            return c[0] + \
                c[1]*x + c[2]*y + c[3]*z + \
                c[4]*x**2 + c[5]*y**2 + c[6]*z**2 + \
                c[7]*x*y + c[8]*x*z + c[9]*y*z + \
                c[10]*x**3 + c[11]*y**3 + c[12]*y**3 + \
                c[13]*x**2*y + c[14]*x**2*z + c[15]*y**2*z + \
                c[16]*y**2*x + c[17]*z**2*x + c[18]*z**2*y + \
                c[19]*x*y*z

        return poly_model(x_den)

    elif dims == 4:

        X = x_sp[:, 0].flatten()
        Y = x_sp[:, 1].flatten()
        Z = x_sp[:, 2].flatten()
        W = x_sp[:, 3].flatten()
        F = f_sp.flatten()

        A = np.array(
            [X*0+1.,
             X, Y, Z, W,
             X**2, Y**2, Z**2, W**2,
             X*Y, X*Z, Y*Z, X*W, Y*W, Z*W,
             X**3, Y**3, Z**3, W**3,
             X*Y*Z, X*Y*W, X*Z*W, Y*W*Z,
             X**2*Y, X**2*Z, X**2*W,
             Y**2*X, Y**2*Z, Y**2*W,
             Z**2*X, Z**2*Y, Z**2*W,
             W**2*X, W**2*Y, W**2*Z]).T

        c, r, rank, s = np.linalg.lstsq(A, F)

        def poly_model(u):
            x = u[:, 0]
            y = u[:, 1]
            z = u[:, 2]
            w = u[:, 3]
            return c[0] + \
                c[1]*x + c[2]*y + c[3]*z + c[4]*w + \
                c[5]*x**2 + c[6]*y**2 + c[7]*z**2 + c[8]*w**2 + \
                c[9]*x*y + c[10]*x*z + c[11]*y*z + \
                c[12]*x*w + c[13]*y*w + c[14]*z*w + \
                c[15]*x**3 + c[16]*y**3 + c[17]*z**3 + c[18]*w**3 + \
                c[19]*x**2*y + c[20]*x**2*z + c[21]*x**2*w + \
                c[22]*y**2*x + c[23]*y**2*z + c[24]*y**2*w + \
                c[25]*z**2*x + c[26]*z**2*y + c[27]*z**2*w + \
                c[28]*w**2*x + c[29]*w**2*y + c[30]*w**2*z


        return poly_model(x_den)

    else:
        print('Error, cannot handle more than 2D')
        return 0

def penalty(g):
    """ Penalty function for constraints in an optimization
        - g: the constraint is such that g <= 0 """

#    return max(0., -1. + float(np.exp(g)))
    return max(0., float(g**1))


def bootstrap(f, vs, num, seed=None):
    """Bootstrapping on the function f with the given samples v_uj
        - f: function to be evaluated
        - vs: list of given samples of underlying uncertainty
        - num: number of times to bootstrap """

    resamples = samplebootstrap(vs, num, seed)
    boot = evalbootstrap(f, resamples, num)

    return boot, resamples


def samplebootstrap(vs, num, seed=None):

    if seed is None:
        np.random.seed(int(time.time()))
    else:
        np.random.seed(seed)

    M = len(vs)
    resamples = []
    for i in range(num):
        vsboot = []
        vrandj = np.floor(np.random.random(M)*M)
        for j, sj in enumerate(vs):
            randj = int(vrandj[j])
            vsboot.append(vs[randj])
        resamples.append(vsboot)

    return resamples


def evalbootstrap(f, mresamples, num):

    boot = np.zeros(num)
    for i in range(num):
        boot[i] = f(mresamples[i])

    return boot


def make_iter(x):
    try:
        iter(x)
        return [xi for xi in x]
    except:
        return [x]

def makeIter(x):
    try:
        iter(x)
        return [xi for xi in x]
    except:
        return [x]


def hm_estimator(vqsamp, targ=None):

    vq = np.sort(vqsamp, kind='mergesort')

    if targ is not None:
        ft = lambda h: targ(h)
    else:
        ft = lambda h: custom_target(h, 0, 0, 5)

    M = len(vqsamp)
    vh = [(1./M)*(0.5 + j) for j in range(M)]
    vt = np.array([float(ft(hi)) for hi in vh])
    Dhat = sum([(1./M)*(vq[j] - vt[j])**2 for j in range(len(vq))])

    Dhat = np.sqrt(Dhat)

    return float(Dhat)


def quantile_estimator(vqsamp, quantile=0.9):

    M = len(vqsamp)
    vq = np.sort(vqsamp, kind='mergesort')
    vh = [(1./M)*(j+1) for j in range(M)]
    for ii, (q, h) in enumerate(zip(vq, vh)):
        if h > quantile:
            Dhat = vq[ii]
            break

#    elif etype.lower() == 'kernels':
#
#        N = self.n_quad
#        M = len(vqsamp)
#
#        vqi = self.vqint.reshape([N, 1])
#        vqj = np.array(vqsamp).reshape([1, M])
#        Kcdf = utils.kernel(vqi - vqj, M, ktype='step')
#        vhi = Kcdf.dot(np.ones([M, 1])).reshape([N,1])
#
#        if vhi[-1] < 0.95:
#            vhi[-1] = 1  # correction for missed sections of CDF.
#        vti = np.array([float(ft(hi)) for hi in vhi]).reshape([N, 1])
#
#        Dhat = self.mat_int(vqi, vhi, vti)

    return float(Dhat)


def getECDF(vqsamp):

    vq = np.sort(vqsamp, kind='mergesort')
    M = len(vqsamp)
    vh = [(1./M)*(0.5 + j) for j in range(M)]
#    vt = np.array([float(ft(hi)) for hi in vh])
#    Dhat = sum([(1./M)*(vq[j] - vt[j])**2 for j in range(len(vq))])
#    Dhat = np.sqrt(Dhat)
    return vq, vh


def getDate():
    return get_date()

def get_date():
    date = datetime.datetime.now()
    months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
              'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    return str(date.day) + months[date.month-1]


def custom_target(x, q0=0, q1=1, p=2):

    if x < 0:
        return q0
    elif x > 1:
        return q1
    else:
        if p > 0:
            return q0 + (q1 - q0)*(x)**p
        else:
            return q1 + (q0 - q1)*(1-x)**abs(p)



