import time
import numpy as np

from include.optim_tools import conversion_pymat as pymat
from include.optim_tools import crafting_phi

from include.optim_tools import CP_covid_4_graph as cp4g


def Rt_PL_graph(dates, data, B_matrix, muR=50, muS=0.005, Gregularization="L1", return_crit = False, Rinit=None):
    """
    Computes the evolution of the reproduction number R for counties on a graph.
    The method used is detailed in optim_tools/CP_covid_4_graph.py (regularized optimization scheme solved using
    Chambolle-Pock algorithm). Hyperparameters choice has to be as followed :
    - mu R sets piecewise linearity of Rt
    - mu S sets spatial regularity of Rt on the chosen graph which transposed incidence matrix is 'B_matrix'.
    :param dates : ndarray of shape (days, )
    :param data : ndarray of shape (counties, days)
    :param B_matrix: ndarray of shape (|E|, counties) : operator matrix for the Graph Total Variations where E are the
    edges of the associated graph. Also corresponds to the transposed incidence matrix
    :param muR: regularization parameter for piecewise linearity of Rt
    :param muS: regularization parameters for spatial coherence
    :return: REstimate : ndarray of shape (days - 1, )
             datesUpdated : list of str of length (days - 1)
             ZDataNorm : ndarray of shape (days - 1) (normalized by county)
             ZPhiNorm : ndarray of shape (days - 1)
             optionals : dict containing execution time, stopping criteria studies
    """
    # Gamma pdf
    Phi = crafting_phi.buildPhi()
    # print("dtype of phi:", Phi.dtype)

    data[data < 0] = 0

    # Normalize each counts for each vertex
    counties, days = np.shape(data)
    ZDataDep = np.zeros((counties, days - 1))
    ZDataNorm = np.zeros((counties, days - 1))
    ZPhiNorm = np.zeros((counties, days - 1))
    datesUpdated = dates[1:]
    for d in range(counties):
        tmpDates, ZDataDep[d], ZPhiDep = crafting_phi.buildZPhi(dates, data[d], Phi)
        # Asserting dates are cropped from first day
        assert (len(tmpDates) == len(datesUpdated))  # == days - 1
        for i in range(days - 1):
            assert (tmpDates[i] == datesUpdated[i])
        # Normalizing for each 'département'
        ZDataNorm[d] = ZDataDep[d] / np.std(ZDataDep[d])
        ZPhiNorm[d] = ZPhiDep / np.std(ZDataDep[d])

    # Run CP covid
    choice = pymat.struct()
    choice.prior = 'laplacian'  # or 'gradient'
    choice.dataterm = 'DKL'  # or 'L2'
    choice.prec = 10 ** (-7)
    choice.nbiterprint = 10 ** 5
    choice.iter = 7 * choice.nbiterprint
    choice.incr = 'R'
    choice.regularization = Gregularization
    if Rinit is not None:
        choice.x0 = Rinit
    
    REstimate, crit, gap, op_out = cp4g.CP_covid_4_graph(ZDataNorm, muR, muS, ZPhiNorm, B_matrix, choice)

    if return_crit:
        return REstimate, datesUpdated, ZDataDep, crit
    else:
        return REstimate, datesUpdated, ZDataDep

