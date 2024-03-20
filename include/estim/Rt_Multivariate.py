import time
from include.optim_tools.Rt_PL_graph import Rt_PL_graph


def Rt_Multivariate(dates, data, B_matrix, muR=50, muS=0.005):
    """
    Computes the evolution of the reproduction number R for the chosen country and between dates 'fday' and 'lday'.
    The method used is detailed in optim_tools/CP_covid_4_graph.py (regularized optimization scheme solved using
    Chambolle-Pock algorithm).
    (optional) One can choose the regularization parameter muR that sets the penalization for piecewise linearity of Rt
    :param dates ndarray of shape (days, )
    :param data ndarray of shape (counties, days)
    :param B_matrix: ndarray of shape (|E|, counties) operator matrix for the Graph Total Variations where E are the
    edges of the associated graph. Also corresponds to the transposed incidence matrix.
    :param muS: regularization parameters for spatial coherence
    :param muR: regularization parameter for piecewise linearity of Rt
    :return: REstimate: ndarray of shape (counties, days - 1), daily estimation of Rt
             timestamps: ndarray of shape (counties, days -1) representing dates
             ZDataDep: ndarray of shape (counties, days - 1) representing processed data
    """
    print("Computing Multivariate estimator ...")
    start_time = time.time()
    REstimate, datesUpdated, ZDataProc = Rt_PL_graph(dates, data, B_matrix, muR, muS)
    executionTime = time.time() - start_time
    print("Done in %.4f seconds ---" % executionTime)
    return REstimate, datesUpdated, ZDataProc
