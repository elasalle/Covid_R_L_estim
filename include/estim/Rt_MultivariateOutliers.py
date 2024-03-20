from include.estim.Rt_Joint_graph import Rt_Jgraph


def Rt_Multivariate(dates, data, B_matrix, lambdaR=3.5, lambdaO=0.02, lambdaS=0.005):
    """
    Computes the evolution of the reproduction number R for the indicated country and between dates 'fday' and 'lday'.
    The method used is detailed in optim_tools/CP_covid_5_outlier_graph.py
    (regularized optimization scheme solved using Chambolle-Pock algorithm).
    Hyperparameters choice has to be as followed :
    - lambda R sets piecewise linearity of Rt
    - lambda O sets sparsity of the outliers Ot
    - lambda S sets total variations regularity on the chosen graph 'G'
    :param dates: list of str of length (days, )
    :param data: ndarray of shape (counties, days)
    :param B_matrix: ndarray of shape (|E|, counties) : operator matrix for the Graph Total Variations where E are the
    edges of the associated graph. Also corresponds to the transposed incidence matrix
    :param lambdaR: regularization parameter for piecewise linearity of Rt
    :param lambdaO: regularization parameters for sparsity of O
    :param lambdaS: regularization parameters for spatial coherence
    :return: REstimate: ndarray of shape (counties, days - 1), daily estimation of Rt
             OEstimate: ndarray of shape (counties, days - 1), daily estimation of Outliers
             timestamps: ndarray of shape (counties, days -1) representing dates
             ZDataDep: ndarray of shape (counties, days - 1) representing processed data
    """
    REstimate, datesUpdated, ZDataProc = Rt_Jgraph(dates, data, B_matrix, lambdaR, lambdaO, lambdaS)
    return REstimate, datesUpdated, ZDataProc
