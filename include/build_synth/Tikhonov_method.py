import numpy as np


def Tikhonov_spat_corr(REstimates, B_matrix, delta):
    """
    Compute diffusion of the signal REstimate on the B_matrix associated to its transposed incidence matrix B_matrix.
    :param REstimates: ndarray of shape (dep, days) R estimates by territory
    :param B_matrix: ndarray of shape (edges, dep) transposed incidence matrix of the associated B_matrix
    :param delta: float hyperparameter controlling the diffusion
    :return: ndarray of shape (dep, days) R estimates diffused
    """
    dep, days = np.shape(REstimates)
    L = np.dot(np.transpose(B_matrix), B_matrix)
    nbChosenDeps, m = np.shape(L)
    assert (m == nbChosenDeps)
    Tikhonov = np.eye(dep) + 2 * delta * L
    return np.linalg.solve(Tikhonov, REstimates)
