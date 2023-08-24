import numpy as np
from include.optim_tools import opL, conversionpymat as pymat, Chambolle_pock_pdm as cppdm, prox_L1 as l1, opLadj, \
    fidelity_terms_DKL as dkl


def set_choice(choice):
    """
    Default choices.
    :param choice: struct (defined in conversionpymat.py).
    :return:
    """
    if not (hasattr(choice, "prior")): choice.prior = "laplacian"
    if not (hasattr(choice, "dataterm")): choice.dataterm = "DKL"
    if not (hasattr(choice, "regularization")): choice.regularization = "L1"
    if not (hasattr(choice, "stop")): choice.stop = "LimSup"
    if not (hasattr(choice, "incr")): choice.incr = "R"

    if not (hasattr(choice, "prec")): choice.prec = 10 ** (-7)
    if not (hasattr(choice, "iter")): choice.iter = 10 ** 7
    if not (hasattr(choice, "nbInf")): choice.nbInf = 10 ** 7
    if not (hasattr(choice, "nbiterprint")): choice.nbiterprint = 10 ** 8

    return


def CP_covid_5_outlier_0cas(data, lambdaR, lambdaO, alpha, choice):
    """
    CP_covid_5_outlier_0cas minimizes the following criterion: if x = [R, O]
    min_{R, O}  L(data, alpha.*u) + lambdaR * Pen1(R) + lambdaO * Pen2(O)
    where L stands either for the Kullback-Leibler divergence or the L2 data term
    and Pen1(R) stands either for the l1 norm applied either on discrete gradient
    for laplacian applied on R, and Pen2 stands for the l1 norm applied on O.
    :param data: ndarray of shape (1, days) : daily new infections (observations)
    :param lambdaR: float : regularization parameter on R (rather discrete gradient of R)
    :param lambdaO: float : regularization parameter on O
    :param alpha: ndarray of shape (days,) data convoluted with infectiousness
    :param choice: structure that stems from MATLAB containing computation options:
        - dataterm: 'DKL' (by default)
        - prec: tolerance for the stopping criterion (1e-7 by default)
        - prior: 'laplacian' (by default)
        - regularization: 'L1' (by default)
    :return: (x, crit, gap, incrR, op_out, objective) such that:
            - x: ndarray of shape (2, days) solution of the minimization problem
            - crit: ndarray of shape (iterations,) values of the objective criterion w.r.t iterations (< choice. iter)
            - gap: ndarray of shape (iterations,) relative difference between the objective criterions of successive
             iterations, maximized on a window of param.stopwin days, if choice.stop = "LimSup".
            - incrR: ndarray of shape (iterations,) relative difference between the normalized estimated R of successive
             iterations, maximized on a window of param.stopwin days, if choice.stop = "LimSup".
            - op_out: for debugging purposes, computes the linear operator without lambdaR and lambdaO contributions
    """
    shape = np.shape(data)
    set_choice(choice)

    if not (hasattr(choice, "x0")):
        choice.x0 = np.array([data, np.zeros(shape)])

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
        cst = np.sum(data[data > 0] * (np.log(data[data > 0]) - 1))
        param.mu = 0

        objective.fidelity = lambda y_, tempdata: dkl.DKLw_outlier(y_, tempdata, alpha) + cst
        prox.fidelity = lambda y_, tempdata, tau: dkl.prox_DKLw_outlier_0cas(y_, tempdata, alpha, tau)
        # # ---------------------------------------------------------------------------------------------
        # # Version where data is already used in objective.fidelity lambda function
        # objective.fidelity = lambda y_: dkl.DKLw_outlier(y_, data, alpha) + cst
        # prox.fidelity = lambda y_, tau: dkl.prox_DKLw_outlier_0cas(y_, data, alpha, tau)
        # # ---------------------------------------------------------------------------------------------

    if choice.regularization == "L1":
        prox.regularization = lambda y_, tau: \
            np.array([l1.prox_L1(y_[0], tau), l1.prox_L1(y_[1], tau), np.maximum(y_[2], np.zeros(np.shape(y_[2])))])
        objective.regularization = lambda y_, tau: tau * np.sum(np.abs(np.concatenate((y_[0], y_[1]))))

    paramL = pymat.struct()
    paramL.lambd = lambdaR
    paramL.type = '1D'
    paramL.op = choice.prior

    op = pymat.struct()

    def direct_covid_5_outlier_0cas(estimates):
        return np.array([opL.opL(estimates[0], filter_def, computation, paramL), lambdaO * estimates[1], estimates[0]])
    op.direct = direct_covid_5_outlier_0cas

    def adjoint_covid_5_outlier_0cas(estimates):
        return np.array([opLadj.opLadj(estimates[0], filter_def, computation, paramL) + estimates[2],
                         lambdaO * estimates[1]])
    op.adjoint = adjoint_covid_5_outlier_0cas

    param.normL = max(lambdaR ** 2 + 1, lambdaO ** 2)

    # data is implicit in objective.fidelity and prox.fidelity
    x, crit, gap = cppdm.PD_ChambollePock_primal_BP(data, param, op, prox, objective)

    op_out = pymat.struct()
    paramL.lambd = 1
    op_out.direct = lambda x_: \
        np.array([opL.opL(x_[0], filter_def, computation, paramL), x_[1], x_[0]])
    op_out.adjoint = lambda x_: \
        np.array([opLadj.opLadj(x_[0], filter_def, computation, paramL) + x_[2], x_[1]])

    return x, crit, gap, op_out, objective

