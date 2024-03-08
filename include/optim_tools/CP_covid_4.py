import numpy as np
from include.optim_tools import opL, conversion_pymat as mat2py, Chambolle_pock_pdm as cppdm, prox_L1 as l1, opLadj, \
    fidelity_terms_DKL as dkl


def set_choice(choice):
    # Default choices
    if not (hasattr(choice, "prior")): choice.prior = "laplacian"
    if not (hasattr(choice, "dataterm")): choice.dataterm = "DKL"
    if not (hasattr(choice, "regularization")): choice.regularization = "L1"
    if not (hasattr(choice, "stop")): choice.stop = "LimSup"
    if not (hasattr(choice, "incr")): choice.incr = 'R'

    if not (hasattr(choice, "prec")): choice.prec = 10 ** (-7)
    if not (hasattr(choice, "nbiterprint")): choice.nbiterprint = 10 ** 5
    if not (hasattr(choice, "iter")): choice.iter = 7 * 10 ** 5
    if not (hasattr(choice, "nbInf")): choice.nbInf = 10 ** 7

    return


def CP_covid_4(data, muR, alpha, choice):
    """
    :param data: ndarray of shape (1,  days) in MATLAB format
    :param muR: float : regularization parameter on R (rather discrete gradient of R)
    :param alpha: ndarray of shape (days,)
    :param choice: structure (see below)
    :return: (x, crit, gap, op_out)

    CP_covid_4 minimizes the following criterion:
    min_u  L(data, alpha.* u) + lambda * R(u)
    where L stands either for the Kullback-Leibler divergence or the L2 data term and R(u) stands either for the l1 norm
    applied either on discrete gradient or laplacian applied on u.

    Input:  - data: observation
            - muR: regularization parameter
            - alpha: ndarray of shape (days,)
            - choice: a structure to select parameters
            - dataterm: 'DKL' (by default)  or 'L2'
            - type: 'usual' (by default) or 'accelerated', the second one is for the strong convex L2
            - prec: tolerance for the stopping criterion (1e-6 by default)
            - prior: 'gradient' (by default) or 'laplacian'
            - regularization: 'L1' (by default) or 'L12'

    Output: - x: solution of the minimization problem
            - crit: values of the objective criterion w.r.t iterations
            - gap: relative difference between the objective criterions of successive iterations
            - op_out: structure containing direct operators for debugging sessions
    """
    set_choice(choice)

    if not (hasattr(choice, "x0")):
        choice.x0 = data

    filter_def = choice.prior
    computation = 'direct'

    param = mat2py.struct()
    param.sigma = 1
    param.tol = choice.prec
    param.iter = choice.iter
    param.stop = choice.stop
    param.nbiterprint = choice.nbiterprint
    param.nbInf = choice.nbInf
    param.x0 = choice.x0
    param.incr = choice.incr
    param.noOutlier = True

    objective = mat2py.struct()
    prox = mat2py.struct()

    if choice.dataterm == "DKL":
        cst = np.sum(data[data > 0] * (np.log(data[data > 0]) - 1))  # WIP
        param.mu = 0
        objective.fidelity = lambda y_, tempData: dkl.DKL_no_outlier(y_, tempData, alpha) + cst
        prox.fidelity = lambda y_, tempData, tau: dkl.prox_DKL_no_outlier(y_, tempData, alpha, tau)

    if choice.regularization == "L1":
        prox.regularization = lambda y_, tau: l1.prox_L1(y_, tau)
        objective.regularization = lambda y_, tau: tau * np.sum(np.abs(y_))

    paramL = mat2py.struct()
    paramL.lambd = muR
    paramL.type = '1D'
    paramL.op = choice.prior

    op = mat2py.struct()
    op.direct = lambda x_: opL.opL(x_, paramL, filter_def, computation)
    op.adjoint = lambda x_: opLadj.opLadj(x_, paramL, filter_def, computation)
    param.normL = muR ** 2  # op.normL(data) # in MATLAB's CP_covid_4 code

    x, crit, gap = cppdm.PD_ChambollePock_primal_BP(data, param, op, prox, objective)

    op_out = mat2py.struct()
    paramL.lambd = 1
    op_out.direct = lambda x_: opL.opL(x_, paramL, filter_def, computation)
    op_out.adjoint = lambda x_: opLadj.opLadj(x_[0], paramL, filter_def, computation)

    return x, crit, gap, op_out