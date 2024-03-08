from include import settings
from include.optim_tools import crafting_phi

# Common libraries for computation
import numpy as np
import time


def Rt_MLE(dates, data):
    """
    Computes the evolution of the reproduction number R for the chosen country and between dates 'fday' and 'lday'
    using the explicit Maximum-Likelihood Estimator
    :param dates ndarray of shape (days, )
    :param data ndarray of shape (days, )
    :return: REstimate : ndarray of shape (days - 1, ), daily estimation of Rt
             OEstimate : ndarray of shape (days - 1, ), daily estimation of Outliers (none here)
             timestamps : ndarray of shape (days -1, )
             ZDataProc : ndarray of shape (days - 1, )
    """
    # Preprocess : ONLY get rid of negative values
    data[data < 0] = 0

    # Compute Phi and convolution Phi * Z (here every vector is cropped from 1 day)
    Phi = crafting_phi.buildPhi(settings.phiBeta, settings.phiAlpha, settings.phiDays)
    timestamps, ZDataProc, ZPhi = crafting_phi.buildZPhi(dates, data, Phi)

    print("Computing Maximum Likelihood Estimator (MLE) ...")
    start = time.time()
    Rt = np.zeros(len(ZDataProc))
    Rt[ZPhi > 0] = ZDataProc[ZPhi > 0] / ZPhi[ZPhi > 0]
    executionTime = time.time() - start
    print("Done in %.4f seconds ---" % executionTime)
    return Rt, timestamps, ZDataProc

# # Choice of country, dates, regularization parameters & computation
# from display import displayFigures
# from estim import getData
# country = 'France'
# fday = '2021-11-01'
# lday = '2022-08-03'
#
# dataBasis = 'JHU'
# NameParameters = 'Fast'
#
# dates, data = getData.getRealData(country, fday, lday, dataBasis)
# REstimateMLE, datesCroppedMLE, dataCroppedMLE = Rt_MLE(dates, data)
#
# dates = datesCroppedMLE
#
# displayFigures.display_MLE(country, dates, REstimateMLE)
