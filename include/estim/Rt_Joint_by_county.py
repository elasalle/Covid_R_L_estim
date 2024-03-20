import numpy as np
from include.estim.Rt_Joint import Rt_J
from include.estim.Rt_Joint_graph import Rt_Jgraph


def Rt_UO_MO(dates, data, B_matrix=np.zeros((1, 1)), lambdaR=3.5, lambdaO=0.02, lambdaS=0.005):
    """
    Computes the spatial and temporal evolution of the reproduction number R and erroneous counts.
    The method used is detailed in optim_tools/CP_covid_5_outlier_graph.py (regularized optimization scheme solved using
    Chambolle-Pock algorithm). Hyperparameters choice has to be as followed :
    - lambda R sets piecewise linearity of Rt
    - lambda O sets sparsity of the outliers Ot
    - lambda S sets total variations regularity on the chosen graph 'G'
    :param dates: list of str of length (days, )
    :param data: ndarray of shape (counties, days)
    :param B_matrix: ndarray of shape (|E|, counties) : operator matrix for the Graph Total Variations where E are the
    edges of the associated graph. Also corresponds to the transposed incidence matrix.
    :param lambdaR: regularization parameter for piecewise linearity of Rt
    :param lambdaO: regularization parameters for sparsity of O
    :param lambdaS: regularization parameters for spatial coherence
    :return: REstimate: ndarray of shape (counties, days - 1), daily estimation of Rt
             OEstimate: ndarray of shape (counties, days - 1), daily estimation of Outliers
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

    edges, counties = np.shape(B_matrix)
    if counties == 1:
        REstimate, OEstimate, datesUpdated, ZDataProc = Rt_J(dates, data, lambdaR, lambdaO)
    else:
        REstimate, OEstimate, datesUpdated, ZDataProc = Rt_Jgraph(dates, data, B_matrix, lambdaR, lambdaO, lambdaS)
    return REstimate, OEstimate, datesUpdated, ZDataProc
