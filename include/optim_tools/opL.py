import numpy as np


def opL(x, filter_def, computation, param):
    """
    Define linear operator (L) associated with the filter in the prior.
    Translation from Nelly Pustelnik matlab's code, CNRS, ENS Lyon June 2019
    :param x: [R, O] with np.shape(R) = np.shape(O) = len(data)
    :param filter_def: option between 'gradient' and 'laplacian'
    :param computation: option between 'direct' and 'fourier'
    :param param: structure with options
    :return: xt <- ndarray of shape np.shape(x)
    """
    dim = np.shape(x)
    xt = np.zeros(dim)

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
                    print("filter_def = %s and computation = %s not implemented yet." % (filter_def, computation))
                else:
                    xt[0][:-2] = param.lambd * (x[0, 2:] / 4 - x[0, 1:- 1] / 2 + x[0, :- 2] / 4)
            else:
                xt = np.dot(np.diag(param.lambd), np.concatenate((x[:, 2:] / 4 - x[:, 1:- 1] / 2 + x[:, :- 2] / 4,
                                                                  np.zeros(dim[0], 2))))
    else:
        OptionError = ValueError("filter_def = %s, computation = %s not implemented yet for type = %s."
                                 % (filter_def, computation, param.type))
        raise OptionError
    return xt