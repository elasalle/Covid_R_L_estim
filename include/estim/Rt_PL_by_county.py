import numpy as np
from include.estim.Rt_PL import Rt_PL
from include.estim.Rt_PL_graph import Rt_PL_graph


def Rt_U_M(dates, data, B_matrix=np.zeros((2, 1)), muR=50, muS=0.005):
    """
    Computes the evolution of the reproduction number R for the chosen country and between dates 'fday' and 'lday'.
    The method used is detailed in optim_tools/CP_covid_4.py (regularized optimization scheme solved using
    Chambolle-Pock algorithm).
    (optional) One can choose the regularization parameter muR that sets the penalization for piecewise linearity of Rt
    :param dates ndarray of shape (days, )
    :param data ndarray of shape (days, )
    :param B_matrix: ndarray of shape (|E|, counties) operator matrix for the Graph Total Variations where E are the
    edges of the associated graph. Also corresponds to the transposed incidence matrix.
    :param muS: regularization parameters for spatial coherence
    :param muR: regularization parameter for piecewise linearity of Rt
    :return: REstimate: ndarray of shape (counties, days - 1), daily estimation of Rt
             timestamps: ndarray of shape (counties, days -1) representing dates
             ZDataDep: ndarray of shape (counties, days - 1) representing processed data
    """
    if len(np.shape(data)) == 1:
        days = len(data)
    elif len(np.shape(data)) == 2:
        dep, days = np.shape(data)
    else:
        ShapeError = TypeError("data should be of shape (days,) or (counties, days) ")
        raise ShapeError
    assert (days == len(dates))

    edges, depG = np.shape(B_matrix)
    if depG == 1:
        REstimate, datesUpdated, ZDataProc = Rt_PL(dates, data, muR)
    else:
        REstimate, datesUpdated, ZDataProc = Rt_PL_graph(dates, data, B_matrix, muR, muS)
    return REstimate, datesUpdated, ZDataProc
