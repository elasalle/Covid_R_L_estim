import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from include.load_data.load_graph_examples import connectStruct_choice, set_Graph_fromMatrix

from include.build_synth import choice_delta as dG
from include.build_synth.Tikhonov_method import Tikhonov_spat_corr
from include.build_synth.compute_spatCorrLevels import compute_spatCorrLevels


colorsCounties = ['#66B2FF', '#006400', '#FFA500', '#FF0000', '#333333']
colorsCountiesDark = ['#0072BD', '#003300', '#994D00', '#990000', '#000000']

plt.rc('text', usetex=True)
plt.rc('text.latex', preamble=r'\usepackage{amsmath}')
plt.style.use('display/matlabMatchingStyles.mplstyle')


def display_connect_structure(example):
    """
    Displays the connectivity structure named 'example'.
    :param example: str to choose the connectivity structure (see include/build_synth/load_graph_examples.py)
    """
    strucMat, pos, labels, colorMap = connectStruct_choice(example)
    G_graph = set_Graph_fromMatrix(strucMat)

    fig, axG = plt.subplots(figsize=(3.75, 2.45))
    fig.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)

    nx.draw(G_graph, ax=axG, labels=labels, node_color=colorMap, pos=pos, node_size=500, font_size=20)
    axG.axis("off")
    fig.show()


def display_spatCorrLevels(R_by_county, B_matrix, fileSuffix='Last'):
    """
    Displays the evolution of the spatial regularization term when considering large logarithmic scale of delta.
    :param R_by_county:
    :param B_matrix:
    :param fileSuffix: (optional) str
    """

    nbDeps, days = np.shape(R_by_county)

    # Computation of deltaGrid through updating the delta_min, delta_max computation
    delta_min, delta_max, powerMin, powerMax = dG.compute_delta_withG(R_by_county, B_matrix, fileSuffix=fileSuffix)

    nbDelta = 500
    if powerMin > 0:
        deltaGrid = np.round(np.logspace(powerMin, powerMax, num=nbDelta), int(powerMin))
    else:
        deltaGrid = np.round(np.logspace(powerMin, powerMax, num=nbDelta), - int(powerMin) + 1)

    RDiff = np.zeros((nbDelta, nbDeps, days))

    spatialRegNorml2 = np.zeros(nbDelta)

    for k in range(len(deltaGrid)):
        delta = deltaGrid[k]
        RDiff[k] = Tikhonov_spat_corr(R_by_county, B_matrix, delta)

        # Spatial regularization term (coordinate-wise norm 2)
        spatialRegNorml2[k] = np.sum(np.abs(np.dot(B_matrix, RDiff[k])) ** 2)

    delta_I, delta_II, delta_III, delta_IV = compute_spatCorrLevels(R_by_county, B_matrix, fileSuffix=fileSuffix)

    lineStyleCorrLevels = {'0': 'solid',
                           'I': 'solid',
                           'II': 'dashed',
                           'III': 'dashdot',
                           'IV': 'dotted'}

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 4.9))
    fig.tight_layout(pad=7.5)
    fig.subplots_adjust(hspace=0.3, left=0.08, right=0.99, bottom=0.22, top=0.88)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    start = len(deltaGrid[deltaGrid < 10 ** (-4)])
    ax.semilogx(deltaGrid[start:], spatialRegNorml2[start:], color='black')
    corrLevel2delta = {'I': delta_I,
                       'II': delta_II,
                       'III': delta_III,
                       'IV': delta_IV}
    for corrLevel in ['I', 'II', 'III', 'IV']:
        ax.vlines([corrLevel2delta[corrLevel]], 0, np.max(spatialRegNorml2),
                  linestyles=lineStyleCorrLevels[corrLevel], label='$\delta_\mathtt{%s}$' % corrLevel, color='black')
    ax.legend(loc=(0.83, 0.075))

    # Formatting the grid
    ax.grid(which="major")
    ax.set_ylabel('$\lVert \mathsf{G}\\boldsymbol{\mathsf{R}}^\star\\rVert^2_2$')
    ax.set_xlabel('$\delta$', labelpad=0, loc='center')
    ax.set_title('Evolution of spatial regularization term')
    fig.show()


def display_spatCorr_onR(R_by_county, B_matrix, deltaList):
    """

    :param R_by_county:
    :param B_matrix:
    :param deltaList: dict
    :return:
    """

    deltaNames = list(deltaList.keys())
    deltaValues = list(deltaList.values())
    assert (len(deltaNames) == len(deltaValues))
    nbDelta = len(deltaValues)
    nbDeps, days = np.shape(R_by_county)
    fig, axes = plt.subplots(nrows=1, ncols=nbDelta, figsize=(35, 4.9))
    fig.tight_layout(pad=7.5)
    fig.subplots_adjust(wspace=0.02, hspace=0.05, left=0.02, right=0.934, bottom=0.22, top=0.86)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    for i in range(nbDelta):
        ax = axes[i]

        delta = deltaValues[i]
        print('Computing diffusion with deltaS = %.2f ----' % delta)
        RDiff = Tikhonov_spat_corr(R_by_county, B_matrix, delta)

        for d in range(nbDeps):
            ax.plot(np.arange(len(RDiff[d])), RDiff[d], label=str(d + 1), color=colorsCounties[d])
        if i == nbDelta - 1:
            ax.legend(loc=(1.05, -0.025), fontsize=35)
            plt.ticklabel_format(style='plain')

        # Formatting the grid
        ax.grid()
        ymax = 1.75
        ax.set_ylim(0.25, ymax + 0.01)
        ax.set_yticks(np.arange(0.5, ymax, 0.5))

        if i != 0:
            labelsY = [item.get_text() for item in ax.get_yticklabels()]
            ax.set_yticklabels([''] * len(labelsY))
            ax.set_title(
                '$\\boldsymbol{\mathsf{R}}^\star(\\boldsymbol{\mathsf{R}}^\dagger ; \delta_\mathtt{%s})$'
                % deltaNames[i])
        else:
            ax.set_title(
                '$\\boldsymbol{\mathsf{R}}^\star(\\boldsymbol{\mathsf{R}}^\dagger ; 0)$')
        ax.set_xlabel('$t$ (days)', labelpad=0, loc='center')

    fig.show()


def display_comparison_uni_multi(example, spatCorrLevel, RrefInit, R_uni, R_multi):
    """

    :param example:
    :param spatCorrLevel:
    :param RrefInit:
    :param R_uni:
    :param R_multi:
    :return:
    """

    counties, days = np.shape(R_uni)
    assert (np.shape(R_multi)[0] == counties)
    assert (np.shape(R_multi)[1] == days)
    assert (np.shape(RrefInit)[0] == counties)
    if np.shape(RrefInit)[1] != days:
        Rref = RrefInit[:, np.shape(RrefInit)[1] - days:]
    else:
        Rref = RrefInit

    REstimates = {'Univariate': R_uni,
                  'Multivariate': R_multi}

    fig, axes = plt.subplots(figsize=(35, 4.9), ncols=counties)
    fig.subplots_adjust(wspace=0.02, hspace=0.01, left=0.02, right=0.996, bottom=0.22, top=0.99)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    for d in range(counties):
        ax = axes[d]
        ax.plot(Rref[d], color=colorsCounties[d], linewidth=5, label='$\\boldsymbol{\mathsf{R}}^\star_%d$' % (d + 1))

        markers_onByDep = np.arange(7, len(REstimates['Univariate'][d]), 15)
        ax.plot(REstimates['Univariate'][d],
                label='$\widehat{\\boldsymbol{\mathsf{R}}}^\mathsf{U}_%d$' % (d + 1),
                color=colorsCounties[d], marker='x', markersize=15, markeredgewidth=4, markevery=markers_onByDep)

        markers_on = np.arange(0, len(REstimates['Multivariate'][d]), 15)

        ax.plot(REstimates['Multivariate'][d],
                label='$\widehat{\\boldsymbol{\mathsf{R}}}^\mathsf{M}_%d$' % (d + 1),
                color=colorsCountiesDark[d], marker='o', markersize=13, markeredgewidth=1, markevery=markers_on)

        plt.ticklabel_format(style='plain')

        ymax = 1.75
        ax.set_ylim(0.25, ymax + 0.01)
        ax.set_yticks(np.arange(0.5, ymax, 0.5))
        ax.grid()

        ax.legend(loc=(0.001, 0.01), fontsize=35)

        if d != 0:
            labelsY = [item.get_text() for item in ax.get_yticklabels()]
            ax.set_yticklabels([''] * len(labelsY))
        else:
            if spatCorrLevel == '0':
                ax.set_ylabel('$\widehat{\\boldsymbol{\mathsf{R}}},\,\delta = %s$' % spatCorrLevel)
            else:
                ax.set_ylabel('$\widehat{\\boldsymbol{\mathsf{R}}},\,\delta_\mathtt{%s}$' % spatCorrLevel)

        ax.set_xlabel('$t$ (days)', labelpad=-1, loc='center')
    # fig.suptitle('Univariate/multivariate estimators for %s connectivity structure' % example)
    fig.show()


def display_comparison_error_uni_multi(example, spatCorrLevel, RrefInit, R_uni, R_multi):
    """

    :param example:
    :param spatCorrLevel:
    :param RrefInit:
    :param R_uni:
    :param R_multi:
    :return:
    """
    counties, days = np.shape(R_uni)
    assert (np.shape(R_multi)[0] == counties)
    assert (np.shape(R_multi)[1] == days)
    assert (np.shape(RrefInit)[0] == counties)
    if np.shape(RrefInit)[1] != days:
        Rref = RrefInit[:, np.shape(RrefInit)[1] - days:]
    else:
        Rref = RrefInit

    REstimates = {'Univariate': R_uni,
                  'Multivariate': R_multi}

    fig, axes = plt.subplots(figsize=(35, 4.9), ncols=counties)
    fig.subplots_adjust(wspace=0.02, hspace=0.01, left=0.02, right=0.996, bottom=0.22, top=0.99)
    plt.ticklabel_format(axis="both", style="sci", scilimits=(0, 0))

    for d in range(counties):
        ax = axes[d]
        markers_onByDep = np.arange(7, len(REstimates['Univariate'][d]), 15)
        ax.plot(np.abs(REstimates['Univariate'][d] - Rref[d]),
                label='$|\widehat{\\boldsymbol{\mathsf{R}}}^\mathsf{U}_%d ' % (d + 1) +
                      '- \\boldsymbol{\mathsf{R}}^\star_%d|$' % (d + 1),
                color=colorsCounties[d], marker='x', markersize=15, markeredgewidth=4, markevery=markers_onByDep)
        markers_on = np.arange(0, len(REstimates['Multivariate'][d]), 15)
        ax.plot(np.abs(REstimates['Multivariate'][d] - Rref[d]),
                label='$|\widehat{\\boldsymbol{\mathsf{R}}}^\mathsf{M}_%d ' % (d + 1) +
                      '- \\boldsymbol{\mathsf{R}}^\star_%d|$' % (d + 1),
                color=colorsCountiesDark[d], marker='o', markersize=13, markeredgewidth=1, markevery=markers_on)

        plt.ticklabel_format(style='plain')

        ax.set_ylim(-0.01, 0.31)
        ax.set_yticks(np.array([0, 0.1, 0.2, 0.3]))
        ax.grid()

        ax.legend(loc=(0.001, 0.475), fontsize=35)

        if d != 0:
            labelsY = [item.get_text() for item in ax.get_yticklabels()]
            ax.set_yticklabels([''] * len(labelsY))
        else:
            if spatCorrLevel == '0':
                ax.set_ylabel('$\widehat{\\boldsymbol{\mathsf{R}}},\,\delta = %s$' % spatCorrLevel)
            else:
                ax.set_ylabel('$\widehat{\\boldsymbol{\mathsf{R}}},\,\delta_\mathtt{%s}$' % spatCorrLevel)

        ax.set_xlabel('$t$ (days)', labelpad=-1, loc='center')

    # fig.suptitle('Error to $\\boldsymbol{\mathsf{R}}^\star$ for %s connectivity structure' % example)
    fig.show()
