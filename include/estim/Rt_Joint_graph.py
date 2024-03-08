import time
import numpy as np

from include.optim_tools import conversion_pymat as pymat
from include.optim_tools import crafting_phi

from include.optim_tools import CP_covid_5_outlier_graph as cp5g


def Rt_Jgraph(dates, data, B_matrix, lambdaR=50, lambdaO=0.02, lambdaS=0.005):
    """
    Computes the evolution of the reproduction number R for the indicated country and between dates 'fday' and 'lday'.
    The method used is detailed in optim_tools/CP_covid_5_outlier_2dep_0cas.py
    (regularized optimization scheme solved using Chambolle-Pock algorithm).
    Hyperparameters choice has to be as followed :
    - lambda R sets piecewise linearity of Rt
    - lambda O sets sparsity of the outliers Ot
    - lambda S sets total variations regularity on the chosen graph 'G'
    :param dates:
    :param data:
    :param B_matrix:
    :param lambdaR:
    :param lambdaO:
    :param lambdaS:
    :return:
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

    # # Run CP covid
    choice = pymat.struct()
    choice.prior = 'laplacian'  # or 'gradient'
    choice.dataterm = 'DKL'  # or 'L2'
    choice.nbiterprint = 10000
    choice.iter = 40000
    choice.nbInf = 7 * choice.nbiterprint
    choice.prec = 10**(-6)
    choice.incr = 'R'

    print("Computing Joint estimation + Graph (JG) ...")
    start_time = time.time()
    xx, crit2, gap, opout = cp5g.CP_covid_5_dpt_outlier_2dep_0cas(ZDataNorm, lambdaR, lambdaS, lambdaO, ZPhiNorm,
                                                                  B_matrix, choice)
    executionTime = time.time() - start_time
    print("Done in %.4f seconds ---" % executionTime)
    REstimate = xx[0]
    OEstimate = xx[1]

    return REstimate, OEstimate, datesUpdated, ZDataNorm, ZPhiNorm, gap

