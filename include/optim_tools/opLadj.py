import numpy as np


def opLadj(y, filter_def, computation, param):
    """
    Computes the adjoint of the operator (L^{*}) used in the penalization.
    Translation from N. PUSTELNIK, CNRS, ENS Lyon, MATLAB code implementation in June 2019.
    :param y: ndarray
    :param filter_def: option between 'gradient' and 'laplacian'
    :param computation: option between 'direct' and 'fourier'
    :param param: structure with options
    :return: x <- ndarray of shape np.shape(y)
    """
    dim = np.shape(y)
    x = np.zeros(dim)

    if "computation" not in locals():
        computation = "direct"
    if "filter_def" not in locals():
        filter_def = "laplacian"

    OptionError = ValueError("filter_def = %s and computation = %s not implemented yet." % (filter_def, computation))

    if param.type == "1D":
        if filter_def == "gradient":
            raise OptionError
        elif filter_def == "laplacian":
            if isinstance(param.lambd, int) or isinstance(param.lambd, float):
                if computation == "fourier":
                    raise OptionError
                else:
                    x[0][0] = 0.25 * y[0, 0]
                    x[0][1] = -0.5 * y[0, 0] + 0.25 * y[0, 1]
                    x[0][2:-2] = 0.25 * y[0, 2:- 2] - 0.5 * y[0, 1:- 3] + 0.25 * y[0, :- 4]
                    x[0][-2] = 0.25 * y[0, - 4] - 0.5 * y[0, - 3]
                    x[0][-1] = 0.25 * y[0, - 3]
                    x[0] = param.lambd * x[0]
            else:
                xbeg = np.array([0.25 * y[0, 0], -0.5 * y[0, 0] + 0.25 * y[0, 1]])
                xend = np.array([0.25 * y[0, - 3] - 0.5 * y[0, - 2], 0.25 * y[0, - 2]])
                x = np.dot(np.diag(param.lambd), np.concatenate((xbeg,
                                                                 0.25 * y[0, 2:- 2] - 0.5 * y[0, 1:- 3] + 0.25 * y[0, :- 4],
                                                                 xend)))
    else:
        raise OptionError
    return x