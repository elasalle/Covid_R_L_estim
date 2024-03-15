import numpy as np


def cropDatesPlusOne(fday, lday, dates, data):
    """
    Crops the data and associated dates between **the day before fday** and lday.
    :param fday: First date ; must be 'year-month-day' format
    :param lday : Last date ; must be 'year-month-day' format or None
    :param dates : ndarray of shape (days,) of dates (object=dtype) for len(dates) = days
    :param data : ndarray of shape (days,) of integers and days == len(data)
    :return : cropDates: ndarray of shape (daysUpdated,) : dates between fday and lday where daysUpdated <= days
              cropData : ndarray of shape (daysUpdated,) : associated data for the same period
    """
    if fday is None:
        first = 0
    else:
        if fday < dates[1] or fday > dates[-1]:
            firstdateErr = ValueError("First day should be between " + dates[1] + " and " + dates[-1])
            raise firstdateErr
        else:
            first = np.argwhere(dates == fday)
            first = first[0, 0] - 1
    if lday is None:
        last = len(dates) - 1
    else:
        if lday < dates[1] or lday > dates[-1]:
            lastdateErr = ValueError("Last day should be between " + dates[1] + " and " + dates[-1])
            raise lastdateErr
        else:
            last = np.argwhere(dates == lday)
            last = last[0, 0]
    cropDates = dates[first:last + 1]
    cropData = data[first:last + 1]
    if fday is None:
        print("Warning : due to initialization and no previous infection counts before %s," % dates[0] +
              " data is exactly %d days long, not %d + 1" % (len(cropDates), len(cropDates)))
    return cropDates, cropData


