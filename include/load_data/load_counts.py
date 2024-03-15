import numpy as np
import pandas as pd
from datetime import date
from collections import OrderedDict

# Loading daily data (essentially for the daily data updates)


def loadingData_byDay():
    """
    (SiDEP data) Loading data from
    https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/
    Available only between May 13th, 2020 and June 27th, 2023
    :return: timestamps: ndarray of str format 'year-month-day' (dates)
             confirmed : ndarray of integers (daily new infections in France)
    """
    # webdata = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/4e8d826a-d2a1-4d69-9ed0-b18a1f3d5ce2', sep=';')
    webdata = pd.read_csv('data/Real-world/SiDEP-France-by-day-2023-06-30-16h26.csv', sep=';')
    # Dates
    timestamps = webdata['jour'].to_numpy()  # str format 'year-month-day'
    confirmedWrongFormat = webdata['P'].to_numpy()
    confirmed = np.array([p.replace(',', '.') for p in confirmedWrongFormat])
    return timestamps, np.array(confirmed, dtype=float)


def loadingData_hosp():
    """
    (SiDEP data) Loading data from
    https://www.data.gouv.fr/fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/
    (daily new hospitalizations per 'départements').
    This data is not maintained since March 31st 2023.
    :return: timestamps: ndarray of str format 'year-month-day' (dates)
             confirmed : ndarray of integers (daily new entrances to the hospital in France)
    """
    # webdata = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7', sep=';')
    webdata = pd.read_csv('data/Real-world/SiDEP-France-hosp-2023-03-31-18h01.csv', sep=';')
    # Dates
    days = webdata['jour'].to_numpy()  # str format 'year-month-day'
    totalDays = date.fromisoformat(days[len(days) - 1]) - date.fromisoformat(days[0])  # datetime format
    totalDays = totalDays.days + 1
    timestamps = days[:totalDays]

    # Data
    nbDepartments = int(len(days) / totalDays)
    hospitalized = webdata['incid_hosp'].to_numpy()
    reanimated = webdata['incid_rea'].to_numpy()
    deaths = webdata['incid_dc'].to_numpy()
    recovered = webdata['incid_rad'].to_numpy()

    H = hospitalized.reshape((nbDepartments, totalDays))  # H[:, i] hospitalized by department
    Rea = reanimated.reshape((nbDepartments, totalDays))  # Rea[i] reanimated by date
    D = deaths.reshape((nbDepartments, totalDays))
    Rec = recovered.reshape((nbDepartments, totalDays))

    totalIncid = H + Rea + D + Rec  # we also add the deaths ?
    confirmed = np.sum(totalIncid, axis=0)  # summing over all 'départements'

    return timestamps, np.array(confirmed, dtype=float)


def loadingData_JHU(country):
    """
    Opens daily new infections for the chosen country, based on JHU data basis.
    Available between January 23th, 2020 and March 9th, 2023.
    :param country : name of the chosen country in str format
    Loading data from Johns Hopkins University (JHU) website containing worldwide daily new infections.
    ( See https://coronavirus.jhu.edu/map.html for more details)
    Processing only data from 'country'.
    :return: timestamps: ndarray of str format 'year-month-day' (dates)
             confirmed : ndarray of integers (daily new infections)
    """
    # url = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/' + \
    #       'csse_covid_19_time_series/JHU-worldwide-covid19-daily-new-infections.csv'
    # webdata = pd.read_csv(url)
    webdata = pd.read_csv('data/Real-world/JHU-worldwide-covid19-daily-new-infections.csv')

    # Dates start at 5th column of webdata columns names
    timestamps = pd.to_datetime(webdata.columns[4:], format='%x').strftime('%Y-%m-%d')  # strftime to get only Y-m-d
    timestamps = timestamps.to_numpy()

    # Get daily new infections
    dataCountries = webdata['Country/Region']
    # dataProvinces = webdata['Province/State']
    arrWebdata = webdata.to_numpy()
    confirmedByCountry = arrWebdata[:, 4:]  # dates start at 5th column of each row
    # provinces = 0
    confirmedAbs = np.zeros(len(timestamps))
    for iC in range(0, len(dataCountries)):
        if dataCountries[iC] == country:
            confirmedAbs = confirmedAbs + confirmedByCountry[iC, :]
            # provinces += 1
    confirmed = np.diff(confirmedAbs)
    timestamps = timestamps[1:]
    return timestamps, np.array(confirmed, dtype=float)


# Loading daily data by 'départements' (returning matrices) ------------------------------------------------------------


def loadingData_hospDep():
    """
    (Loading data from
    https://www.data.gouv.fr/fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/
    (daily new hospitalizations by French 'département')
    Will mostly be used in graph version, which is still WIP.
    This data is not maintained since March 31st 2023.
    :return: timestamps: ndarray of str format 'year-month-day' (dates)
             confirmed : ndarray matrix of integers (daily new entrances to the hospital in France) by 'département'
                         of shape (totalDeps, totalDays)
    """
    # webdata = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/63352e38-d353-4b54-bfd1-f1b3ee1cabd7', sep=';')
    webdata = pd.read_csv('data/Real-world/SiDEP-France-hosp-2023-03-31-18h01.csv', sep=';')

    # Dates
    days = webdata['jour'].to_numpy()  # str format 'year-month-day'
    totalDays = date.fromisoformat(days[len(days) - 1]) - date.fromisoformat(days[0])  # datetime format
    totalDays = totalDays.days + 1  # integer now
    timestamps = days[:totalDays]

    # Retrieving infection counts
    nbDepartments = int(len(days) / totalDays)
    hospitalized = webdata['hosp'].to_numpy()
    reanimated = webdata['rea'].to_numpy()
    deaths = webdata['dc'].to_numpy()
    recovered = webdata['rad'].to_numpy()

    H = hospitalized.reshape((nbDepartments, totalDays))  # H[:, i] hospitalized by department
    Rea = reanimated.reshape((nbDepartments, totalDays))  # Rea[i] reanimated by date
    D = deaths.reshape((nbDepartments, totalDays))
    Rec = recovered.reshape((nbDepartments, totalDays))

    totalIncid = H + Rea + D + Rec  # we also add the deaths ?
    confirmed = totalIncid

    return np.array(timestamps), np.array(confirmed, dtype=float)


def loadingData_byDep():
    """
    (SiDEP data) Loading data from
    https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/
    Daily new infections per 'département' for 102 'départements' (not considering 977 and 978)
    Will mostly be used in graph version, which is still WIP.
    Note : data is sorted daily month by month and the first month starts on May 13th 2020.
    :return: timestamps: ndarray of str format 'year-month-day' (dates from 2020-05-13)
             confirmed : ndarray matrix of integers (daily new entrances to the hospital in France) by 'département'
    """
    # webdata = pd.read_csv('https://www.data.gouv.fr/fr/datasets/r/426bab53-e3f5-4c6a-9d54-dba4442b3dbc', sep=';')
    webdata = pd.read_csv('data/Real-world/SiDEP-France-by-day-by-dep-2023-06-30-16h26.csv', sep=';')
    # Total number of days
    days = webdata['jour'].to_numpy()  # str format 'year-month-day'
    totalDays = date.fromisoformat(days[-1]) - date.fromisoformat(days[0])  # datetime format
    totalDays = totalDays.days + 1  # last day included
    timestamps = list(OrderedDict.fromkeys(days))
    assert(totalDays == len(timestamps))
    assert(timestamps[0] == days[0])
    assert(timestamps[-1] == days[- 1])

    # Total number of 'Départements'
    depsRaw = webdata['dep'].to_numpy()
    allDeps = list(OrderedDict.fromkeys(depsRaw))[:-2]  # cropping the two last indexes : 977 and 977
    totalDeps = len(allDeps)
    assert(totalDeps == np.max(np.shape(np.where(days == days[-1]))) - 2)

    # Retrieving infection counts
    confirmed = np.zeros((totalDeps, totalDays))
    allInfectionsWrongFormat = webdata['P'].to_numpy()
    allInfections = np.array([p.replace(',', '.') for p in allInfectionsWrongFormat])
    for indexDep in np.arange(totalDeps):
        confirmed[indexDep] = allInfections[np.where(depsRaw == allDeps[indexDep])]
    return np.array(timestamps), np.array(confirmed, dtype=float), allDeps
