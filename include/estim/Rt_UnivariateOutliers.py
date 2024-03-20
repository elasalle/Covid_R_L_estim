import numpy as np

from include.optim_tools import conversion_pymat as pymat
from include.estim.Rt_Joint_graph import Rt_Jgraph


def Rt_Univariate_Outliers(dates, data, lambdaR=3.5, lambdaO=0.02):
    """
    Computes the spatial and temporal evolution of the reproduction number R and erroneous counts.
    Can be used for time series.
    The method used is detailed in optim_tools/CP_covid_5_outlier_graph.py (regularized optimization scheme solved using
    Chambolle-Pock algorithm) but with neither spatial regularization nor explicit underlying connectivity structure.
     Hyperparameters choice has to be as followed :
    - lambda R sets piecewise linearity of Rt
    - lambda O sets sparsity of the outliers Ot
    :param dates ndarray of shape (days, )
    :param data ndarray of shape (counties, days) or (days, )
    :param lambdaR: regularization parameter for piecewise linearity of Rt
    :param lambdaO: regularization parameters for sparsity of O
    :return: REstimate: ndarray of shape (counties, days - 1), daily estimation of Rt
             datesUpdated: ndarray of shape (counties, days -1) representing dates
             dataCrop: ndarray of shape (counties, days - 1) representing processed data
    """
    if len(np.shape(data)) == 1:
        days = len(data)
        counties = 1
        dataProc = pymat.pyvec2matvec(data)
    elif len(np.shape(data)) == 2:
        counties, days = np.shape(data)
        dataProc = data
    else:
        ShapeError = TypeError("data should be of shape (days,) or (counties, days) ")
        raise ShapeError
    assert (days == len(dates))

    B_matrix = np.zeros((2, counties))
    REstimate, OEstimate, datesUpdated, dataCrop = Rt_Jgraph(dates, dataProc, B_matrix, lambdaR, lambdaO, lambdaS=0)
    if len(np.shape(data)) == 1:
        assert (np.shape(REstimate)[0] == 1)
        assert (np.shape(REstimate)[1] == days - 1)
        return pymat.matvec2pyvec(REstimate), pymat.matvec2pyvec(OEstimate), datesUpdated, pymat.matvec2pyvec(dataCrop)
    return REstimate, OEstimate, datesUpdated, dataCrop
