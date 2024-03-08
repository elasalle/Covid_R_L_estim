import numpy as np
import scipy.stats as spst

from include.optim_tools import opL
from include.optim_tools import conversion_pymat as pymat

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


def JaccardIndexREstim(R1, R2):
    """
    Computes the Jaccard index (in %) between two R estimates slope changes of same length.
    More precisely, computes the Jaccard index between discrete laplacian operators associated to each time serie.
    :param R1: ndarray of shape (len(R1),)
    :param R2: ndarray of shape (len(R2),) where len(R2) = len(R1)
    :return: float : Jaccard index
    """
    paramL1 = pymat.struct()
    paramL1.lambd = 1  # just a factor
    paramL1.type = '1D'
    paramL1.op = 'laplacian'
    D2R_opDirect = lambda x_: opL.opL(x_, 'laplacian', 'direct', paramL1)

    assert (len(R1)) == np.max(len(R2))

    laplacianR1 = pymat.matvec2pyvec(D2R_opDirect(pymat.pyvec2matvec(R1)))
    laplacianR2 = pymat.matvec2pyvec(D2R_opDirect(pymat.pyvec2matvec(R2)))

    laplacianR1[np.abs(laplacianR1) < 10 ** (-3)] = 0
    laplacianR2[np.abs(laplacianR2) < 10 ** (-3)] = 0

    D2Rs = np.zeros((2, len(R1)))
    D2Rs[0] = np.abs(laplacianR1)
    D2Rs[1] = np.abs(laplacianR2)

    JaccardIndex, Intersection, Union = JaccardIndexSignal(D2Rs)
    return JaccardIndex * 100


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
