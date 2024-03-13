from include import settings
from include.optim_tools import conversion_pymat as pymat, CP_covid_5_outlier as cp50, crafting_phi

# Common libraries for computation
import numpy as np
import time


def Rt_J(dates, data, lambdaR=3.5, lambdaO=0.03):
    """
    Computes the evolution of the reproduction number R for the indicated country and between dates 'fday' and 'lday'.
    The method used is detailed in optim_tools/CP_covid_5_outlier.py (regularized optimization scheme solved using
    Chambolle-Pock algorithm). Hyperparameters choice has to be as followed :
    - lambda R sets piecewise linearity of Rt
    - lambda O sets sparsity of the outliers Ot
    :param dates: list of str of length (days, )
    :param data: ndarray of shape (days, )
    :param lambdaR: regularization parameter for piecewise linearity of Rt
    :param lambdaO: regularization parameters for sparsity of O
    :return: REstimate: ndarray of shape (days - 1, ), daily estimation of Rt
             OEstimate: ndarray of shape (days - 1, ), daily estimation of Outliers
             timestamps: ndarray of shape (days -1, ) representing dates
             ZDataProc: ndarray of shape (days - 1, ) representing processed data
    """
    assert(len(data) == len(dates))

    # Preprocess : ONLY get rid of negative values
    data[data < 0] = 0

    # Compute Phi and convolution Phi * Z (here every vector is cropped from 1 day)
    Phi = crafting_phi.buildPhi(settings.phiBeta, settings.phiAlpha, settings.phiDays)
    timestamps, ZDataProc, ZPhi = crafting_phi.buildZPhi(dates, data, Phi)
    ZDataProc = pymat.pyvec2matvec(ZDataProc)

    # ----------------------------------------------------------------------------------------------------------
    # Run Chambolle-Pock algorithm for R0 estimation
    choice = pymat.struct()
    choice.prior = 'laplacian'  # or 'gradient'
    choice.dataterm = 'DKL'  # or 'L2'
    choice.nbiterprint = 10 ** 5
    choice.iter = 7 * choice.nbiterprint
    choice.nbInf = 7 * choice.nbiterprint
    choice.prec = 10 ** (-6)
    choice.incr = 'R'

    choice.x0 = np.array([np.ones(np.shape(ZDataProc)), np.zeros(np.shape(ZDataProc))])

    print("Computing joint estimation (J) ...")
    start_time = time.time()
    xx, crit, gap, op_out, objective = cp50.CP_covid_5_outlier_0cas(ZDataProc / np.std(ZDataProc),
                                                                    lambdaR, lambdaO,
                                                                    ZPhi / np.std(ZDataProc), choice)
    executionTime = time.time() - start_time
    print("Done in %s seconds ---" % executionTime)

    xr = xx[0]
    xo = xx[1] * np.std(ZDataProc)

    return pymat.matvec2pyvec(xr), pymat.matvec2pyvec(xo), timestamps, pymat.matvec2pyvec(ZDataProc)
