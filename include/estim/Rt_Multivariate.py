import time
import numpy as np
from include.optim_tools.Rt_PL_graph import Rt_PL_graph


def Rt_M(data, muR=50, muS=0.005, options=None):
    """
    Computes the evolution of the reproduction number R for the chosen country and between dates 'fday' and 'lday'.
    The method used is detailed in optim_tools/CP_covid_4_graph.py (regularized optimization scheme solved using
    Chambolle-Pock algorithm).
    (optional) One can choose the regularization parameter muR that sets the penalization for piecewise linearity of Rt
    :param data ndarray of shape (counties, days)
    :param muS: regularization parameters for spatial coherence
    :param muR: regularization parameter for piecewise linearity of Rt
    :param options: dictionary containing 'dates', 'B_matrix'
    :return: REstimate: ndarray of shape (counties, days - 1), daily estimation of Rt
             timestamps: ndarray of shape (counties, days -1) representing dates
             ZDataDep: ndarray of shape (counties, days - 1) representing processed data
    """
    dates = options['dates']
    B_matrix = options['B_matrix']
    print("Computing Multivariate estimator ...")
    start_time = time.time()
    REstimate, datesUpdated, ZDataProc = Rt_PL_graph(dates, data, B_matrix, muR, muS)
    executionTime = time.time() - start_time
    print("Done in %.4f seconds ---" % executionTime)

    if 'counties' in list(options.keys()):
        options_M = {'dates': datesUpdated,
                     'data': ZDataProc,
                     'counties': options['counties']}
    else:
        options_M = {'dates': datesUpdated,
                     'data': ZDataProc,
                     'counties': [str(i) for i in range(np.shape(data)[0])]}
    return REstimate, options_M
