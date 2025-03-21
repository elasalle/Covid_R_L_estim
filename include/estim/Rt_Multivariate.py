import time
import numpy as np
from include.optim_tools.Rt_PL_graph import Rt_PL_graph


def Rt_M(data, muR=50, muS=0.005, options=None, Gregularization="L1"):
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
             options: dictionary containing at least:
             - dates: ndarray of shape (counties, days -1) representing dates
             - data: ndarray of shape (counties, days - 1) representing processed data
    """
    dates = options['dates']
    B_matrix = options['B_matrix']
    print("Computing Multivariate estimator ...")
    start_time = time.time()
    REstimate, datesUpdated, ZDataProc = Rt_PL_graph(dates, data, B_matrix, muR, muS, Gregularization)
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

def Rt_with_laplacianReg(data, L, muR=50, muS=0.005, Gregularization="L2", dates=None, verbose=False, return_crit=False, Rinit = None):

    if verbose:
        print("Computing a Cholesky decomposition of L ...")
    ti = time.time()
    try:
        S = np.linalg.cholesky(L + (1e-12)*np.eye(L.shape[0]))
    except:
        minegval = np.linalg.eigvalsh(L)[0]
        print("minimal eignvalue of L was {}".format(minegval))
        S = np.linalg.cholesky(L - (1.01*minegval)*np.eye(L.shape[0]))
    tf = time.time()
    if verbose:
        print("Done in {:4.3f} seconds ---".format(tf-ti))

    
    start_time = time.time()
    REstimate, datesUpdated, ZDataProc, crits = Rt_PL_graph(dates, data, S.T, muR, muS, Gregularization, return_crit=True, Rinit=Rinit)
    executionTime = time.time() - start_time
    if verbose:
        print("Multivariate estimator computed in %.4f seconds ---" % executionTime)
    if return_crit:
        return REstimate, crits[-1]
    else:
        return REstimate
