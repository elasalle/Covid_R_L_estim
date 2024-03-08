import numpy as np


def SignaltoNoiseRatio(groundTruth, estimation):
    """
    Computes the Signal-to-Noise-Ratio (SNR in dB) which stands for the quadratic error between
    the ground truth and estimation.
    :param groundTruth: ndarray of shape (deps, days) -- or (days,)
    :param estimation: ndarray of shape (deps, days) -- or (days,)
    :return: SNR(groundTruth, estimation)
    """
    normOrder = 2
    assert (len(groundTruth) == len(estimation))
    SquaredError = np.sum(np.abs(estimation - groundTruth) ** normOrder)
    return 10 * np.log10(np.sum(np.abs(groundTruth) ** normOrder) / SquaredError)


def SignaltoNoiseRatioMC(groundTruth, estimations):
    """
    Compute the Signal-to-Noise-Ratio (SNR in dB) for multiple draws in Monte-Carlo simulations, which R estimations are
    gathered in 'estimations'.
    Returns meanSNR, errorSNR such that SNR(estimations) = meanSNR +/- errorSNR.
    :param groundTruth: ndarray of shape (deps, days) --- or (days,)
    :param estimations: ndarray of shape (nbDraws, deps, days) --- or (nbDraws, days)
    :return: meanSNR: float
             errorSNR: float
    """
    nbDraws, days = np.shape(estimations)
    assert (days == len(groundTruth))

    SNREstim = np.zeros(nbDraws)
    for draw in range(nbDraws):
        SNREstim[draw] = SignaltoNoiseRatio(groundTruth, estimations[draw])
    return (1 / nbDraws) * np.sum(SNREstim), (1.96 / np.sqrt(nbDraws)) * np.std(SNREstim)
