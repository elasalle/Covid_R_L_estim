import time
import numpy as np

from include.optim_tools import conversion_pymat as pymat
from include.optim_tools import crafting_phi

from include.optim_tools import CP_covid_4_graph as cp4g


def Rt_PL_graph(dates, data, B_matrix, muR=50, muS=0.008):
    """
    Computes the evolution of the reproduction number R for the indicated country and between dates 'fday' and 'lday'.
    The method used is detailed in optim_tools/CP_covid_4_graph.py
    (regularized optimization scheme solved using Chambolle-Pock algorithm).
    Hyperparameters choice has to be as followed :
    - mu R sets piecewise linearity of Rt
    - mu S sets total variations regularity on the chosen graph 'G'
    :param dates : ndarray of shape (days, )
    :param data : ndarray of shape (days, )
    :param B_matrix: ndarray of shape (nbEdges, nbTerritories ) matrix encoding the graph used
    :param muR: float
    :param muS: float
    :return: REstimate : ndarray of shape (days - 1, )
             datesUpdated : list of str of length (days - 1)
             ZDataNorm : ndarray of shape (days - 1) NORMALIZED
             ZPhiNorm : ndarray of shape (days - 1)
             optionals : dict containing execution time, stopping criteria studies
    """
    # Gamma pdf
    Phi = crafting_phi.buildPhi()

    # Normalize each counts for each vertex
    depts, days = np.shape(data)
    ZDataDep = np.zeros((depts, days - 1))
    ZPhiDep = np.zeros((depts, days - 1))
    ZDataNorm = np.zeros((depts, days - 1))
    ZPhiNorm = np.zeros((depts, days - 1))
    datesUpdated = dates[1:]
    for d in range(depts):
        tmpDates, ZDataDep[d], ZPhiDep[d] = crafting_phi.buildZPhi(dates, data[d], Phi)
        # Asserting dates are cropped from first day
        assert (len(tmpDates) == len(datesUpdated))  # == days - 1
        for i in range(days - 1):
            assert (tmpDates[i] == datesUpdated[i])
        # Normalizing for each 'département'
        ZDataNorm[d] = ZDataDep[d] / np.std(ZDataDep[d])
        ZPhiNorm[d] = ZPhiDep[d] / np.std(ZDataDep[d])

    # Run CP covid
    choice = pymat.struct()
    choice.prior = 'laplacian'  # or 'gradient'
    choice.dataterm = 'DKL'  # or 'L2'
    choice.prec = 10 ** (-7)
    choice.nbiterprint = 10 ** 5
    choice.iter = 7 * choice.nbiterprint
    choice.incr = 'R'

    print("Computing Penalized Log-likelihood + Graph (PLG) ...")
    start_time = time.time()
    REstimate, crit, gap, op_out = cp4g.CP_covid_4_graph(ZDataNorm, muR, muS, ZPhiNorm, B_matrix, choice)
    executionTime = time.time() - start_time
    print("Done in %.4f seconds ---" % executionTime)

    optionals = {'executionTime': executionTime,
                 'crit': crit,
                 'gap': gap,
                 'prec': choice.prec}

    return REstimate, datesUpdated, ZDataNorm, ZPhiNorm, optionals

