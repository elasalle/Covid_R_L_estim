from scipy.io import loadmat


def loadROconfig(configuration, returnFirstDay=False):
    """
    Load R O as ndarray, available in data/.
    :param configuration: str between 'I', 'II', 'III', 'IV'
    :param returnFirstDay: bool to get
    return: Rref: ndarray of shape (days, )
            Oref: ndarray of shape (days, )
            firstDay (optional): str in format 'YYYY-MM-DD'
    """
    inputData = loadmat('data/Synthetic/Univariate/Config_%s.mat' % configuration, squeeze_me=True)
    assert(inputData['configuration'] == configuration)
    if returnFirstDay:
        return inputData['Rref'], inputData['OutliersRef'], inputData['firstDay']
    else:
        return inputData['Rref'], inputData['OutliersRef']
