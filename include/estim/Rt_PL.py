from include import settings
from include.optim_tools import conversion_pymat as pymat, CP_covid_4 as cp4, crafting_phi

# Common libraries for computation
import numpy as np
import time


def Rt_PL(dates, data, muR=50):
    """
    Computes the evolution of the reproduction number R for the chosen country and between dates 'fday' and 'lday'.
    The method used is detailed in optim_tools/CP_covid_4.py (regularized optimization scheme solved using
    Chambolle-Pock algorithm).
    (optional) One can choose the regularization parameter muR that sets the penalization for piecewise linearity of Rt
    :param dates: list of str of length (days, )
    :param data ndarray of shape (days, )
    :param muR : regularization parameter for piecewise linearity of Rt
    :return: REstimate : ndarray of shape (days - 1, ), daily estimation of Rt
             timestamps : ndarray of shape (days -1, )
             ZDataProc : ndarray of shape (days - 1, ) not normalized !
    """
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
    choice.nbiterprint = 100000
    choice.iter = 7 * choice.nbiterprint
    choice.nbInf = 7 * choice.nbiterprint
    choice.prec = 10 ** (-7)
    choice.incr = 'R'

    choice.x0 = np.ones(np.shape(ZDataProc))

    print("Computing Penalized Log-likelihood (PL) ...")
    start_time = time.time()
    xx, crit, gap, opout = cp4.CP_covid_4(ZDataProc / np.std(ZDataProc), muR, ZPhi / np.std(ZDataProc), choice)
    executionTime = time.time() - start_time
    print("Done in %.4f seconds ---" % executionTime)

    xr = xx[0]

    return pymat.matvec2pyvec(xr), timestamps, pymat.matvec2pyvec(ZDataProc)
