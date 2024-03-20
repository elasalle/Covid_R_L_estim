import numpy as np
from include.load_data import date_choice, load_counts as load


def get_real_counts(country, fday, lday, dataBasis):
    """
    Returns real dates and associated new daily Covid19 cases data in the adequate format,
    between the day before fday (for initialization) and lday. Data is opened from either dataBasis='SPF'
    Santé Publique France or 'JHU' Johns Hopkins University.
    :param country: str between all countries available
    :param fday: str in format 'YYYY-MM-DD'
    :param lday: str in format 'YYYY-MM-DD'
    :param dataBasis: str between 'SPF' and 'JHU'. See ./load_counts.py
    :return: dates ndarray of shape (days + 1, ) of str in format 'YYYY-MM-DD'
             data  ndarray of shape (days + 1, ) of float (round numbers)
    """
    # Opening data with chosen country
    if dataBasis == 'JHU':
        print("Opening data from Johns Hopkins University.")
        timestampsInit, ZDataInit = load.loadingData_JHU(country)
    elif dataBasis == 'SPF':
        if country == 'France':
            print("Opening data from Santé Publique France.")
            timestampsInit, ZDataInit = load.loadingData_byDay()
        else:
            CountryError = ValueError("Santé Publique France (SPB) only provides data for France, not %s." % country)
            raise CountryError
    else:
        DataBasisUnknown = ValueError("Data Basis %s unknown." % dataBasis)
        raise DataBasisUnknown

    # Crop to dates choice
    timestampsCropped, ZDataCropped = date_choice.cropDatesPlusOne(fday, lday, timestampsInit, ZDataInit)

    return timestampsCropped, ZDataCropped


def get_real_counts_by_county(fday, lday, dataBasis, chosenDep='all'):
    """
    :param fday:
    :param lday:
    :param dataBasis:
    :param chosenDep: either str 'all', or list of chosen 'départements' ex : [38, 69] of length 'dep'
    :return: dates ndarray of shape (days + 1, ) of str in format 'YYYY-MM-DD'
             data  ndarray of shape (dep, days + 1) of float (round numbers)
    """
    if dataBasis == 'SPF':
        timestampsInit, ZDataDepInit, allDeps = load.loadingData_byDep()
    elif dataBasis == 'hosp':
        timestampsInit, ZDataDepInit, allDeps = load.loadingData_hospDep()
    else:
        DataBasisUnknown = ValueError("Data Basis %s unknown." % dataBasis)
        raise DataBasisUnknown

    # Cropping following dates (time cropping)
    timestampsCropped, ZDataDepCropped = date_choice.cropDatesPlusOne(fday, lday, timestampsInit, ZDataDepInit)

    deps = np.array(allDeps[:96])  # spatial cropping to remove DROM-COM

    # Cropping following spatial sorting
    if chosenDep == 'all':
        print("Selecting following areas : %s" % deps)
        return timestampsCropped, ZDataDepCropped[:96], deps
    else:
        print("Selecting following areas : %s" % deps[chosenDep])
        return timestampsCropped, ZDataDepCropped[chosenDep], deps[chosenDep]
