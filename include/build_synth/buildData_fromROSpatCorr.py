import time
import numpy as np

from include.load_data.graphExamples import deps

from include.build_synth.load_RByDeps import loadRByDeps_onGraph
from include.build_synth.Tikhonov_method import Tikhonov_spat_corr
from include.build_synth import buildData_fromRO


def buildDataMulti_ROSpatCorr(period, example, deltaS, nbDraws, spat_corr='Tikhonov'):
    """
    Rebuild new data from R and 0 estimation using Rt_Joint applied on each 'département' on new daily infections.
    Using the function buildDataFromRO.buildDataMulti_anyRO which computes data for each 'département' as the following
    [data Z drawn from Poisson distribution with mean (R * Phi Z + Out)].
    :param period: str studied dates
    :param example: str between 'Line', 'Sun' etc. see graphExamples.py
    :param deltaS: float regularization parameter over diffusion : float
    :param nbDraws: int deciding of the number of Poisson draws
    :param spat_corr: str equals to 'Tikhonov' by default.
    :return: Rref : ndarray of shape (days,), OutliersRef : ndarray of shape (days,),
             ZDataInit : ndarray of shape (days,), ZDataBuilt : ndarray of shape (days,)
    """

    dates, ZDataInit, R_by_county, Outliers_by_county, B_matrix, G_graph, optionals = loadRByDeps_onGraph(period, example)
    assert (len(deps) == len(optionals['deps']))

    nbDeps, days = np.shape(R_by_county)
    assert (len(dates) == days)
    assert (len(deps) == nbDeps)

    # Adding inter-county correlations to R_by_county to get spatially regularized ground truth R ----
    if spat_corr == 'Tikhonov':
        spat_corr_func = Tikhonov_spat_corr
    else:
        diffusionOptErr = ValueError("diffusionOpt not implemented yet. Try 'Tikhonov' or 'l2penl1'.")
        raise diffusionOptErr

    print('Introducing inter-county correlation using %s regularization with deltaS = %.2f.' % (spat_corr, deltaS))
    start_time = time.time()
    R_spat_corr = spat_corr_func(R_by_county, B_matrix, deltaS)
    executionTime = time.time() - start_time
    print("Done in %s seconds ---" % executionTime)

    # Correcting first cases and cropping ground truth from first day
    firstCases, RByDeps_crop, OutliersRef = \
        buildData_fromRO.firstCasesCorrection(ZDataInit, R_by_county, Outliers_by_county)
    Rref = R_spat_corr[:, 1:]  # here Rref is the ground truth that is spatially diffused.
    assert(np.shape(Rref)[1] == np.shape(OutliersRef)[1])

    # Building synthetic infection counts by county -----------------------------------------------------------------------
    datesBuilt = []
    ZDataBuilt = np.zeros((nbDraws, nbDeps, days - 1))
    for draw in range(nbDraws):
        datesBuilt, ZDataBuilt[draw] = buildData_fromRO.buildDataMulti_anyRO(Rref, OutliersRef, firstCases)

    assert (len(datesBuilt) == days - 1)
    return datesBuilt, ZDataBuilt, Rref, OutliersRef, B_matrix, G_graph
