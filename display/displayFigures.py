import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from display import formattingFigures as format


def display_data(dates, data, title, savefig=False, savePath=None):
    """
    Display daily new infections data and associated dates and given title.
    :param dates: ndarray of shape (days, ) for str in format 'YYYY-MM-DD'
    :param data: ndarray of shape (days, ) float of daily new infections
    :param title: str title
    :param savefig: (optional) bool for saving figure or not
    :param savePath: (optional) if savefig is True, precise the path where the figure is saved.
    :return: fig, ax and dates (in matplotlib.dates format)
    """
    formattedDates = [mdates.datestr2num(t) for t in dates]
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19.5, 9.5))
    fig.tight_layout(pad=5.0)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    ax.plot(formattedDates, data, label="$\mathsf{Z}_t$", color='black')
    ax.set(title=title, ylabel='$New cases \mathsf{Z}_t$')

    # Formatting the grid
    ax.grid(which="major", linestyle='-', alpha=0.6)
    ax.grid(which="minor", linestyle='--', alpha=0.3)
    ax.legend()

    # Formatting xticks
    locator, dateFormatter = format.adaptiveDaysLocator(formattedDates)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(dateFormatter)

    # Formatting yticks
    ax.set_ylim(0, format.adaptiveYLimit(data))
    yScalarFormatter = format.ScalarFormatterClass(useMathText=True)
    yScalarFormatter.set_powerlimits((0, 0))
    ax.yaxis.set_major_formatter(yScalarFormatter)

    if savefig:
        fig.savefig(savePath)
    fig.show()
    return fig, ax, formattedDates


def display_REstim(dates, data, REstimate, method, OEstimate=None, comparison=False, RTrue=None,
                   dataUnder=False, savefig=False, savePath=None):
    """
    Display an estimation of R and Outliers estimations.
    :param dates: ndarray of shape (days, ) for str in format 'YYYY-MM-DD'
    :param data: ndarray of shape (days, ) float
    :param REstimate: ndarray of shape (days, ) float
    :param method: str between 'MLE', 'Cori', 'PL' and 'Joint' method to estimate R.
    :param OEstimate: (optional) ndarray of shape (days, )
    :param RTrue: (optional) ndarray of shape (days, )
    :param dataUnder: (optional) bool to display used data below on another axis. Can't be used if displaying RTrue
    :param comparison: (optional) bool to display multiple R estimations. method shoud be a list indexing REstimate.
    :param savefig: (optional) bool for saving figure or not
    :param savePath: (optional) if savefig is True, precise the path where the figure is saved.
    :return: fig, ax and dates (in matplotlib.dates format)
    """
    formattedDates = [mdates.datestr2num(t) for t in dates]
    if dataUnder:
        fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(19.5, 12))
        fig.tight_layout(pad=6.5)
        fig.subplots_adjust(hspace=0.3, right=0.99, bottom=0.05, top=0.95)
        axR = axes[1]
        ax = axes[0]
    else:
        fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(19.5, 9.5))
        fig.tight_layout(pad=5)
        axR = ax
    
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    # Displaying (or not) data used for following estimation ---
    if dataUnder:
        ax.plot(formattedDates, data, label="$\mathsf{Z}$", color='black')
        if comparison:
            for m in ['PL', 'Joint']:
                ax.plot(formattedDates, data - OEstimate[m], label='$\mathsf{Z}^{\mathsf{denoised}}$ ($\mathsf{%s}$)' % m,
                        color=format.colors['denoised%s' % m])
        elif OEstimate is not None and method in ['PL', 'Joint']:
            ax.plot(formattedDates, data - OEstimate, label='$\mathsf{Z}^{\mathsf{denoised}}$ ',
                    color=format.colors['denoised%s' % method])
        ax.legend(loc='upper left')
        ax.set(ylabel='New cases $\mathsf{Z}_t$')
    else:
        if OEstimate is not None:
            NoDisplayData = ValueError("Cannot display estimated outliers if no data displayed." +
                                       " Please set dataUnder = True")
            raise NoDisplayData

    # Displaying R estimation(s) ---
    if comparison:
        for m in method:
            axR.plot(formattedDates, REstimate[m], label='$\mathsf{R}^{\mathsf{%s}}$' % m, color=format.colors[m])
        axR.legend(loc='lower left')
    else:
        axR.plot(formattedDates, REstimate, label='$\mathsf{R}^{\mathsf{%s}}$' % method, color=format.colors[method])
        axR.legend(loc='lower left')

    # Displaying ground truth RTrue (if available) ---
    if RTrue is not None:
        if len(RTrue) != len(data):
            assert (len(RTrue) > len(data))
            RTrueCropped = RTrue[len(RTrue) - len(data):]
        else:
            RTrueCropped = RTrue
        axR.plot(formattedDates, RTrueCropped, color='black', label='$\mathsf{R}^{\mathrm{true}}$')

    axR.set(ylabel='$\mathsf{R}_t$')

    # Formatting xticks ---
    locator, dateFormatter = format.adaptiveDaysLocator(formattedDates)

    axR.xaxis.set_major_locator(locator)
    axR.xaxis.set_major_formatter(dateFormatter)

    # Formatting yticks ---
    yScalarFormatter = format.ScalarFormatterClass(useMathText=True)
    yScalarFormatter.set_powerlimits((0, 0))
    axR.set_ylim(0, 2)
    axR.set_yticks([0, 0.5, 1, 1.5, 2])

    # Formatting the grid ---
    axR.grid(which="major", linestyle='-', alpha=0.6)
    axR.grid(which="minor", linestyle='--', alpha=0.3)

    if dataUnder:
        # Formatting xticks ---
        ax.xaxis.set_major_locator(locator)
        ax.xaxis.set_major_formatter(dateFormatter)
        # Formatting yticks ---
        ax.set_ylim(0, format.adaptiveYLimit(data))
        ax.yaxis.set_major_formatter(yScalarFormatter)
        # Formatting the grid ---
        ax.grid(which="major", linestyle='-', alpha=0.6)
        ax.grid(which="minor", linestyle='--', alpha=0.3)
        ax.set_xticklabels([])

    # Saving figure (or not) ---
    if savefig:
        fig.savefig(savePath)
    fig.show()
    return fig, ax, formattedDates


def display_dataBuilt(datesBuilt, dataBuilt, RTrue, OTrue, displayO=False,
                      savefig=False, savePath=''):
    """
    Display the generated daily new cases (dataBuilt) compared to the ground truth (RTrue, OTrue) used to do so.
    :param datesBuilt: ndarray of shape (days, ) for str in format 'YYYY-MM-DD'
    :param dataBuilt: ndarray of shape (days, )
    :param RTrue: ndarray of shape (days, )
    :param OTrue: ndarray of shape (days, )
    :param displayO: (optional) bool for displaying OTrue or not
    :param savefig: (optional) bool for saving figure or not
    :param savePath: (optional) if savefig is True, precise the path where the figure is saved.
    :return: fig, ax and dates (in matplotlib.dates format)
    """
    formattedDates = [mdates.datestr2num(t) for t in datesBuilt]
    fig, axes = plt.subplots(nrows=2, ncols=1, figsize=(19.5, 12))
    fig.tight_layout(pad=7.5)
    fig.subplots_adjust(hspace=0.3, right=0.99, bottom=0.05, top=0.95)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    axR = axes[0]

    axR.plot(formattedDates, RTrue, color='black', label='$\mathsf{R}^{\mathrm{true}}$')
    axR.set(ylabel='$\mathsf{R}_t$')
    axR.legend(loc='lower left')

    ax = axes[1]
    ax.plot(formattedDates, dataBuilt, label="$\mathsf{Z}$", color=format.colors['synthData'])
    if displayO:
        ax.plot(formattedDates, dataBuilt - OTrue, label="$\mathsf{Z} - \mathsf{O}$", color='black')
    ax.legend(loc='upper left')
    ax.set(ylabel='New cases $\mathsf{Z}_t$')

    # Formatting xticks
    locator, dateFormatter = format.adaptiveDaysLocator(formattedDates)
    axR.xaxis.set_major_locator(locator)
    axR.xaxis.set_major_formatter(dateFormatter)

    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(dateFormatter)

    # Formatting yticks
    yScalarFormatter = format.ScalarFormatterClass(useMathText=True)
    yScalarFormatter.set_powerlimits((0, 0))

    axR.set_ylim(0, 2)
    axR.set_yticks([0, 0.5, 1, 1.5, 2])

    ax.set_ylim(0, format.adaptiveYLimit(dataBuilt))
    ax.yaxis.set_major_formatter(yScalarFormatter)

    # Formatting the grid
    axR.grid(which="major", linestyle='-', alpha=0.6)
    axR.grid(which="minor", linestyle='--', alpha=0.3)
    # Formatting the grid 2nd figure
    ax.grid(which="major", linestyle='-', alpha=0.6)
    ax.grid(which="minor", linestyle='--', alpha=0.3)
    ax.set_xticklabels([])

    # ax.set_title(method)
    if savefig:
        fig.savefig(savePath)
    fig.show()
    return fig, ax, formattedDates
