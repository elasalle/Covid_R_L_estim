from include.load_data import datesChoice, loadingData as load


def getRealData(country, fday, lday, dataBasis):
    """
    Returns real dates and associated new daily Covid19 cases data in the adequate format,
    between the day before fday (for initialization) and lday. Data is opened from either dataBasis='SPF'
    Santé Publique France or 'JHU' Johns Hopkins University.
    :param country: str between all countries available
    :param fday: str in format 'YYYY-MM-DD'
    :param lday: str in format 'YYYY-MM-DD'
    :param dataBasis: str between 'SPF' and 'JHU'. See ./loadingData.py
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
    timestampsCropped, ZDataCropped = datesChoice.cropDatesPlusOne(fday, lday, timestampsInit, ZDataInit)

    return timestampsCropped, ZDataCropped


