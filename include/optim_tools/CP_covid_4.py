import numpy as np
from include.optim_tools import opL, conversionpymat as pymat, Chambolle_pock_pdm as cppdm, prox_L1 as l1, opLadj, \
    fidelity_terms_DKL as dkl


def set_choice(choice):
    # Default choices
    if not (hasattr(choice, "prior")): choice.prior = "laplacian"
    if not (hasattr(choice, "dataterm")): choice.dataterm = "DKL"
    if not (hasattr(choice, "regularization")): choice.regularization = "L1"
    if not (hasattr(choice, "stop")): choice.stop = "LimSup"
    if not (hasattr(choice, "incr")): choice.incr = 'R'

    if not (hasattr(choice, "prec")): choice.prec = 10 ** (-7)
    if not (hasattr(choice, "iter")): choice.iter = 10 ** 7
    if not (hasattr(choice, "nbInf")): choice.nbInf = 10 ** 7
    if not (hasattr(choice, "nbiterprint")): choice.nbiterprint = 10 ** 8

    return


def CP_covid_4(data, muR, alpha, choice):
    """
    CP_covid_4 minimizes the following criterion:
    min_u  L(data, alpha.* u) + lambda * R(u)
    where L stands either for the Kullback-Leibler divergence or the L2 data term
    and R(u) stands either for the l1 norm applied either on discrete gradient or laplacian applied on u.
    :param data: ndarray of shape (1,  days)
    :param muR: float : regularization parameter on R (more precisely on discrete gradient of R)
    :param alpha: ndarray of shape (days,) data convoluted with infectiousness
    :param choice: structure that stems from MATLAB containing computation options:
        - dataterm: 'DKL' (by default)
        - prec: tolerance for the stopping criterion (1e-7 by default)
        - prior: 'laplacian' (by default)
        - regularization: 'L1' (by default)
    :return: (x, crit, gap, op_out) such that:
            - x: ndarray of shape (2, days) solution of the minimization problem
            - crit: ndarray of shape (iterations,) values of the objective criterion w.r.t iterations (< choice. iter)
            - gap: ndarray of shape (iterations,) relative difference between the objective criterions of successive
             iterations. If choice.stop = "LimSup", this quantity is maximized on a window of param.stopwin days.
             Else it is computed in L2 norm.
            - op_out: for debugging purposes, computes the linear operator without lambdaR and lambdaO contributions
    """
    set_choice(choice)

    if not (hasattr(choice, "x0")):
        choice.x0 = data

    filter_def = choice.prior
    computation = 'direct'

    param = pymat.struct()
    param.sigma = 1
    param.tol = choice.prec
    param.iter = choice.iter
    param.stop = choice.stop
    param.nbiterprint = choice.nbiterprint
    param.nbInf = choice.nbInf
    param.x0 = choice.x0
    param.incr = choice.incr

    objective = pymat.struct()
    prox = pymat.struct()

    if choice.dataterm == "DKL":
        cst = np.sum(data[data > 0] * (np.log(data[data > 0]) - 1))  # WIP
        param.mu = 0
        objective.fidelity = lambda y_, tempData: dkl.DKL_no_outlier(y_, tempData, alpha) + cst
        prox.fidelity = lambda y_, tempData, tau: dkl.prox_DKL_no_outlier(y_, tempData, alpha, tau)

    if choice.regularization == "L1":
        prox.regularization = lambda y_, tau: l1.prox_L1(y_, tau)
        objective.regularization = lambda y_, tau: tau * np.sum(np.abs(y_))

    paramL = pymat.struct()
    paramL.lambd = muR
    paramL.type = '1D'
    paramL.op = choice.prior

    op = pymat.struct()
    op.direct = lambda x_: opL.opL(x_, filter_def, computation, paramL)
    op.adjoint = lambda x_: opLadj.opLadj(x_, filter_def, computation, paramL)
    param.normL = muR ** 2 + 1  # op.normL(data) # in MATLAB's CP_covid_4 code

    x, crit, gap = cppdm.PD_ChambollePock_primal_BP(data, param, op, prox, objective)

    op_out = pymat.struct()
    paramL.lambd = 1
    op_out.direct = lambda x_: opL.opL(x_, filter_def, computation, paramL)
    op_out.adjoint = lambda x_: opLadj.opLadj(x_[0], filter_def, computation, paramL)

    return x, crit, gap, op_out