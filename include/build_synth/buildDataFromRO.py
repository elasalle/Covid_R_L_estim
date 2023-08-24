import numpy as np

import settings
from include.optim_tools import craftingPhi as phi

import pandas as pd


def randomDates(firstDay, days):
    """
    Returns a list of dates in format 'YYYY-MM-DD' from firstDay on until 'days' days later.
    :param firstDay: str in format 'YYYY-MM-DD'
    :param days: integer of days starting from firstDay. Also equals to len(dates).
    :return: dates: list of str in format 'YYYY-MM-DD'
    """
    randomDays = pd.date_range(firstDay, periods=days, freq='D')
    dates = [day.strftime("%Y-%m-%d") for day in randomDays]
    return dates


def buildData_anyRO(R, Outliers, firstCases, firstDay='2020-01-23', threshold=settings.thresholdPoisson):
    """
    Build data Z drawn from Poisson distribution with mean (R * Phi Z + Out) with the given firstCases (1 day)
    :param R: ndarray of shape (days,)
    :param firstCases : number of cases of the original data, only used for initialization.
    :param Outliers: ndarray of shape (days, )
    :param firstDay: (optional) str in format 'YYYY-MM-DD' to indicate first day of random dates drawn
    :param threshold: (optional) inferior limit for Poisson parameter : float
    :return datesBuilt: list of str in format 'YYYY-MM-DD' random dates associated with ZDataBuilt
            ZDataBuilt: ndarray of shape (days, ) built following Cori's epidemiological model
    """
    days = len(R)  # should be one day less than the original data it was computed from
    assert (days > 26)
    assert(days == len(Outliers))

    Phi = phi.buildPhi()

    # Initialization with known values for the first day only
    ZData = np.zeros(days + 1)
    ZData[0] = firstCases
    # scale = firstCases / (np.max(Outliers[0]) * 150)  # WIP to ensure outliers and firstCases are on the same scale
    scale = 1
    OutliersRescaled = scale * Outliers
    realR = np.ones(days + 1)  # realR[0] not relevant since it will be cropped out

    tauPhi = len(Phi) - 1  # wrong explanation in associated papers (tauPhi = len(Phi) -1 = 25)
    # Modified convolution : normalized convolution for the first tauPhi days.
    for k in range(1, tauPhi):
        daysIterK = len(ZData[:k + 1])
        assert (daysIterK > 1)  # if there's only one day of data, can not compute ZPhi
        PhiNormalizedIterK = Phi[1:k + 1] / np.sum(Phi[1:k + 1])  # careful to use non-normalized Phi here !!
        fZ = np.flip(ZData[:k])  # 1st value of Phi is always 0 : we do not need data on day 0
        realR[k] = R[k-1] * np.sum(fZ * PhiNormalizedIterK)  # R is already cropped of day 1
        ZData[k] = np.random.poisson(max(realR[k] + OutliersRescaled[k-1], threshold))   # Outliers are cropped of day 1

    PhiNormalized = Phi / np.sum(Phi)
    for k in range(tauPhi, days + 1):
        fZ = np.flip(ZData[k - len(Phi) + 1:k])
        realR[k] = R[k-1] * np.sum(fZ * PhiNormalized[1:])  # 1st value of Phi is always 0 : we don't need data on day 0
        ZData[k] = np.random.poisson(max(realR[k] + OutliersRescaled[k-1], threshold))

    return randomDates(firstDay, len(ZData[1:])), ZData[1:]  # cropped from the initialization with 'real' firstCases
