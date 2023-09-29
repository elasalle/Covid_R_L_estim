import settings
from include.optim_tools import craftingPhi

# Common libraries for computation
import numpy as np
import time


def Rt_C(dates, data, tau=settings.tauWindow):
    """
    Computes the evolution of the reproduction number R for the chosen country and between dates 'fday' and 'lday'
    using Cori's method (C) using a prior assuming that Rt is constant on time periods of length 'tau' days.
    :param dates ndarray of shape (days, )
    :param data ndarray of shape (days, )
    :param tau : (optional) integer, number of days for which prior distribution is supposed piecewise constant
    :return: REstimate : ndarray of shape (days - 1, ), daily estimation of Rt
             OEstimate : ndarray of shape (days - 1, ), daily estimation of Outliers (none here)
             timestamps : ndarray of shape (days -1, )
             ZDataProc : ndarray of shape (days - 1, )
    """
    # Preprocess : ONLY get rid of negative values
    data[data < 0] = 0

    # Compute Phi and convolution Phi * Z (here every vector is cropped from 1 day)
    Phi = craftingPhi.buildPhi(settings.phiBeta, settings.phiAlpha, settings.phiDays)
    timestamps, ZDataProc, ZPhi = craftingPhi.buildZPhi(dates, data, Phi)
    days = len(ZDataProc)
    Rt = np.zeros(days)
    print("Computing Cori's method estimation (C) ...")
    start = time.time()
    for t in range(1, days):
        posteriorA = settings.priorA + np.sum(ZDataProc[max(t - tau + 1, 0):t + 1])
        posteriorB = 1 / settings.priorB + np.sum(ZPhi[max(t - tau + 1, 0):t + 1])
        if posteriorB > 0:
            Rt[t] = posteriorA / posteriorB
        else:
            Rt[t] = 0
    executionTime = time.time() - start
    print("Done in %.4f seconds ---" % executionTime)
    return Rt, timestamps, ZDataProc
