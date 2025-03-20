
import numpy as np

from include.estim.Rt_UnivariateOutliers import Rt_U_O
from include.estim.Rt_Univariate import myRt_U
from include.estim import Rt_Multivariate as RtM
from include.estim import laplacianLearning as LL
from include.estim import Rt_MLE as RtMLE

from include.optim_tools.fidelity_terms_DKL import DKL_no_outlier as DKL
from include.optim_tools import crafting_phi,  opL, conversion_pymat as mat2py

Phi = crafting_phi.buildPhi()

def get_normalized_Zphi_and_Z(Z):
    nb_deps, days = Z.shape
    ZDataDep  = np.zeros((nb_deps, days - 1))
    ZPhiDep   = np.zeros((nb_deps, days - 1))
    ZPhiNorm  = np.zeros((nb_deps, days - 1))
    ZDataNorm = np.zeros((nb_deps, days - 1))

    for d in range(nb_deps):
        _, ZDataDep[d], ZPhiDep[d] = crafting_phi.buildZPhi(None, Z[d], Phi)
       
    # Normalizing for each 'département'
    std = np.std(ZDataDep, axis=1)
    ZDataNorm = ZDataDep / std[:,None]
    ZPhiNorm = ZPhiDep / std[:,None]
    return ZDataNorm, ZPhiNorm


def obj_function(R, L, ZDataNorm, ZPhiNorm, lambda_pwlin, lambda_GR, lambda_Fro):

    cst = np.sum(ZDataNorm[ZDataNorm > 0] * (np.log(ZDataNorm[ZDataNorm > 0]) - 1))
    KL_term = DKL(R, ZDataNorm, ZPhiNorm) + cst

    param_pwlin = mat2py.struct()
    param_pwlin.lambd = lambda_pwlin
    param_pwlin.type = '1D'
    pwlin_term = np.sum(np.abs(opL.opL(R, param_pwlin)))

    GR_term = lambda_GR * np.sum(R * (L @ R))

    Fro_term = lambda_Fro * np.sum(L**2)

    crit = KL_term + pwlin_term + GR_term + Fro_term
    crit_L = GR_term + Fro_term
    crit_R = KL_term + pwlin_term + GR_term

    return crit, crit_L, crit_R

def make_lambda_GR_as_list(max_iter, lambda_GR):
    #make sure that lambda_GR is a list of size max_iter
    if isinstance(lambda_GR, list):
        if len(lambda_GR)<max_iter:
            lambda_GR = lambda_GR + [lambda_GR[-1]]*(max_iter - len(lambda_GR)) #if the initial list is to small, complete with the last value
        else:
            lambda_GR = lambda_GR[:max_iter] #if it is too long, cut it.
    elif isinstance(lambda_GR, (float, int)):
        lambda_GR = [lambda_GR]*max_iter # create a constant list
    else:
        ValueError("lambda_GR should be a list, an int or a float, received {}.".format(type(lambda_GR)))
    return lambda_GR

def initialize_alternate_optim(Z, ndep, options, init_method="U", init_param=None):
    if init_method=="U":
        if init_param is None:
            init_param = {"options":options, "lambdaU_pwlin":50}
        R = myRt_U(Z,  init_param["lambdaU_pwlin"], init_param["options"])
    elif init_method=="MLE":
        if init_param is None:
            init_param = {"options":options}
        R = []
        for i in range(ndep):
            Ri, _ = RtMLE.Rt_MLE(Z[i], init_param["options"])
            R.append(Ri)
        R = np.array(R)
    elif init_method=="UO":
        if init_param is None:
            init_param = {"options":options, "lambdaU_pwlin":3.5, "lambdaU_O":0.02}
        R, _, _ = Rt_U_O(Z, init_param["lambdaU_pwlin"], init_param["lambdaU_O"], init_param["options"])
    return R


def Rt_L(Z, max_iter,lambda_pwlin, lambda_GR, lambda_Fro, options, init_method="U", init_param=None, save_objective=False):

    ndep = Z.shape[0]
    
    #handle param
    Gregularization="L2"
    dates = options["dates"]

    lambda_GR = make_lambda_GR_as_list(max_iter, lambda_GR)
    
    #initialize variables        
    Restims = []
    Lestims = []

    if save_objective:
        ZDataNorm, ZPhiNorm = get_normalized_Zphi_and_Z(Z) # we compute once here. it will be usefull to compute the objective function at each iteration
    objs = []
    crits_R, crits_L = [], []
    crits_R_true, crits_L_true = [], []


    # initialize
    R = initialize_alternate_optim(Z, init_method, init_param)
    Restims.append(R)

    L = - np.ones((ndep, ndep)) / (ndep-1)
    np.fill_diagonal(L, 1)
    Lestims.append(L)

    if save_objective:
        objs.append(obj_function(R,L,ZDataNorm, ZPhiNorm, lambda_pwlin, lambda_GR[-1], lambda_Fro)[0])

    for iter, lambda_gr in enumerate(lambda_GR):
        print("lambda_GR = {:5.3f}".format(lambda_gr))    
        L, crit_L_true = LL.learningL(lambda_gr, lambda_Fro, R, return_crit=True)
        if save_objective:
            _, crit_L, _ = obj_function(R,L,ZDataNorm, ZPhiNorm, lambda_pwlin, lambda_GR[-1], lambda_Fro)
            crits_L.append(crit_L)
            crits_L_true.append(crit_L_true)

        R, crit_R_true = RtM.Rt_with_laplacianReg(Z, L, lambda_pwlin, lambda_gr, Gregularization, dates, return_crit=True)
        if save_objective:
            crit, _, crit_R = obj_function(R,L,ZDataNorm, ZPhiNorm, lambda_pwlin, lambda_GR[-1], lambda_Fro)
            objs.append(crit)
            crits_R.append(crit_R)
            crits_R_true.append(crit_R_true)
        
        Lestims.append(L)
        Restims.append(R)

    return Restims, Lestims, lambda_GR, objs, crits_R, crits_R_true, crits_L, crits_L_true