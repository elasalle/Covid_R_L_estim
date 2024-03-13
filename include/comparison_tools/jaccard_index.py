import numpy as np
import scipy.stats as spst

from include.optim_tools.opL import opL

winGaussRef = 5  # to capture similarities over 2 days before and after, in Jaccard index
sigmaGaussRef = 1.2


def JaccardIndexSignal(X, winGauss=winGaussRef, sigmaGauss=sigmaGaussRef):
    """
    Computes the Jaccard index between two binary sets that are convoluted with a Gaussian function of size winGauss and
    standard deviation sigmaGauss, beforehand.
    :param X: ndarray of shape (2, days) containing 2 sets of the same size : days
    :param winGauss: integer (in days) of window size of the Gaussian, should be odd and positive
    :param sigmaGauss: float for the standard deviation of the Gaussian
    :return: JaccardIndex = Intersection/Union : float : Jaccard index,
             Intersection : sum of float
             Union : sum of float
    """
    # Stands for fspecial('gaussian', [1 winGauss], sigmaGauss) in MATLAB ----------------------------------------------
    larg = (winGauss - 1) / 2
    GaussianLowpassFilter = spst.norm(0, sigmaGauss).pdf(np.arange(- larg, larg + 1))
    GaussianLowpassFilter = GaussianLowpassFilter / np.sum(GaussianLowpassFilter)
    # ------------------------------------------------------------------------------------------------------------------
    # print('Gaussian Lowpass filter used :', GaussianLowpassFilter)
    numberOfSets, days = np.shape(X)
    XGauss = np.zeros((numberOfSets, days + winGauss - 1))  # nb of days is because we do not troncate after convolution
    for k in range(numberOfSets):
        XGauss[k] = np.convolve(X[k], GaussianLowpassFilter)  # not causal convolution here
        # XGauss[k] = XGausstmp[:days]  # not used : troncation to get the same number of days than initially

    IntersectionsGauss = np.sqrt(XGauss[0] * XGauss[1])
    Intersection = np.sum(IntersectionsGauss)
    UnionsGauss = XGauss[0] + XGauss[1] - IntersectionsGauss
    Union = np.sum(UnionsGauss)
    return Intersection / Union, Intersection, Union


def JaccardIndexREstim(R1true, R2true):
    """
    Computes the Jaccard index (in %) between two R estimates slope changes of same length.
    More precisely, computes the Jaccard index between discrete laplacian operators associated to each time serie.
    :param R1true: ndarray of shape (len(R1),) or (deps, days)
    :param R2true: ndarray of shape (len(R2),) where len(R2) = len(R1) or (deps, days)
    :return: float : Jaccard index
    """

    if len(np.shape(R1true)) == 1:
        R1 = np.reshape(R1true, (1, len(R1true)))
        R2 = np.reshape(R2true, (1, len(R2true)))
    else:
        R1 = R1true
        R2 = R2true

    nbDeps, days = np.shape(R1)
    assert (np.shape(R2)[0] == nbDeps)
    assert (np.shape(R2)[1] == days)

    laplacianR1 = opL(R1)
    laplacianR2 = opL(R2)

    laplacianR1[np.abs(laplacianR1) < 10 ** (-3)] = 0
    laplacianR2[np.abs(laplacianR2) < 10 ** (-3)] = 0

    JaccardIndex = np.zeros(nbDeps)
    Intersection = np.zeros(nbDeps)
    Union = np.zeros(nbDeps)
    D2Rs = np.zeros((nbDeps, 2, days))
    for k in range(nbDeps):
        D2Rs[k, 0] = np.abs(laplacianR1[k])
        D2Rs[k, 1] = np.abs(laplacianR2[k])
        JaccardIndex[k], Intersection[k], Union[k] = JaccardIndexSignal(D2Rs[k])

    if len(np.shape(R1true)) == 1:
        return np.array(JaccardIndex.flatten()) * 100
    else:
        return np.array(JaccardIndex) * 100


def JaccardIndexAveraged(groundTruth, estimations):
    nbDeps, days = np.shape(groundTruth)
    assert (np.shape(estimations)[0] == nbDeps)
    assert (np.shape(estimations)[1] == days)

    JaccardByDep = np.zeros(nbDeps)
    for d in range(nbDeps):
        JaccardByDep[d] = JaccardIndexREstim(groundTruth[d], estimations[d])  # 1D version of computing Jaccard index

    return 1 / nbDeps * np.sum(JaccardByDep)  # averaging over territories


def JaccardIndexREstimMC(groundTruth, estimations):
    """
    Compute the Jaccard index (previously defined) for multiple draws in Monte-Carlo simulations,
    which R estimations are gathered in 'estimations'.
    Returns meanJaccard, errorJaccard such that JaccardIndex(estimations) = meanJaccard +/- errorJaccard.
    :param groundTruth: ndarray of shape (days, )
    :param estimations: ndarray of shape (nbDraws, days)
    :return: meanJaccard: float
             errorJaccard: float
    """
    nbDraws, days = np.shape(estimations)
    assert (days == len(groundTruth))

    JaccardIndexEstim = np.zeros(nbDraws)
    for d in range(nbDraws):
        JaccardIndexEstim[d] = JaccardIndexREstim(groundTruth, estimations[d])
    return (1 / nbDraws) * np.sum(JaccardIndexEstim) * 100, (1.96 / np.sqrt(nbDraws)) * np.std(JaccardIndexEstim) * 100


