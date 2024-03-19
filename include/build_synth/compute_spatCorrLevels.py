from scipy.io import savemat, loadmat
from include.build_synth import choice_delta as dG


def compute_spatCorrLevels(R_by_county, B_matrix, fileSuffix='Last', saveData=True):
    """
    :param R_by_county:
    :param B_matrix:
    :param fileSuffix:
    :param saveData:
    :return:
    """

    # Computation delta_min, delta_max computation
    delta_min, delta_max, powerMin, powerMax = \
        dG.compute_delta_withG(R_by_county, B_matrix, fileSuffix=fileSuffix)

    fileInit = loadmat("data/Synthetic/Multivariate/Line_graph/spatCorrLevels_Line.mat", squeeze_me=True)
    delta_I = fileInit['deltaSLow']/fileInit['deltaSmin'] * delta_min
    delta_II = fileInit['deltaSMedium'] / fileInit['deltaSmin'] * delta_min
    delta_III = fileInit['deltaSHigh'] / fileInit['deltaSmin'] * delta_min
    delta_IV = fileInit['deltaSVeryHigh'] / fileInit['deltaSmin'] * delta_min

    if saveData:
        savemat("include/build_synth/deltaFiles/chosenSpatCorrLevels%s.mat" % fileSuffix,
                {'delta_I': delta_I,
                 'delta_II': delta_II,
                 'delta_III': delta_III,
                 'delta_IV': delta_IV,
                 'B_matrix': B_matrix,
                 'R_by_county': R_by_county,
                 'name': fileSuffix})

    return delta_I, delta_II, delta_III, delta_IV


