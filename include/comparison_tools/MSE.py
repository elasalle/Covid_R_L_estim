import numpy as np


def MSE_oneDraw(groundTruth, estimation):
    """
    Computes the Squared Error between the ground truth and estimation.
    :param groundTruth: ndarray of shape (deps, days) -- or (days,)
    :param estimation: ndarray of shape (deps, days) -- or (days,)
    :return: SquaredError(groundTruth, estimation)
    """
    normOrder = 2
    assert (len(groundTruth) == len(estimation))
    return np.sum(np.abs(estimation - groundTruth) ** normOrder)


def MSEByDep_indic(groundTruth, estimation):
    """
    :param estimation: ndarray of shape (nbDeps, days)
    :param groundTruth: ndarray of shape (nbDeps, days)
    :return:
    """
    nbDeps, days = np.shape(estimation)
    assert (nbDeps == np.shape(groundTruth)[0])
    assert (days == np.shape(groundTruth)[1])

    indicators = np.zeros(nbDeps)
    for d in range(nbDeps):
        indicators[d] = MSE_oneDraw(estimation[d], groundTruth[d])

    return indicators  # we should minimize this criteria


def MSENormByDep_indic(groundTruth, estimation):
    """
    :param estimation: ndarray of shape (nbDeps, days)
    :param groundTruth: ndarray of shape (nbDeps, days)
    :return:
    """
    nbDeps, days = np.shape(estimation)
    assert (nbDeps == np.shape(groundTruth)[0])
    assert (days == np.shape(groundTruth)[1])

    indicators = np.zeros(nbDeps)
    for d in range(nbDeps):
        indicators[d] = MSE_oneDraw(estimation[d], groundTruth[d]) / np.sum(groundTruth[d] ** 2)
    return indicators * 100  # we should minimize this criteria


def MSEMeanNorm_indic(groundTruth, estimation):
    """
    :param estimation: ndarray of shape (nbDeps, days)
    :param groundTruth: ndarray of shape (nbDeps, days)
    :return:
    """
    nbDeps, days = np.shape(estimation)
    assert (nbDeps == np.shape(groundTruth)[0])
    assert (days == np.shape(groundTruth)[1])

    indicatorsByDep = MSEByDep_indic(groundTruth, estimation)
    assert (len(indicatorsByDep) == nbDeps)
    groundTruthNormByDep = np.sum(groundTruth ** 2, axis=1)

    indicators = 1 / nbDeps * np.sum(indicatorsByDep / groundTruthNormByDep)
    assert ((indicators - 1 / nbDeps * np.sum(MSENormByDep_indic(groundTruth, estimation))) < 10 ** (-16))

    return indicators * 100  # percentage; we should minimize this criteria