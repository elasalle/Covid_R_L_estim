import numpy as np

from include.load_data.get_counts import get_real_counts
from include.estim.Rt_UnivariateOutliers import Rt_U_O
from include.build_synth import buildData_fromRO as build

def generate_synthZ(countries, cluster_sizes, firstDay, lastDay, alpha=None, lambdaU_L = 3.5, lambdaU_O = 0.03):
    nclusters = len(cluster_sizes)
    
    # get number of new cases for each cluster (i.e. each country)
    ZData_by_cluster = []
    for i in range(nclusters):
        ZData, options = get_real_counts(countries[i], firstDay, lastDay, 'JHU')
        ZData_by_cluster.append(ZData)
    ZData_by_cluster = np.array(ZData_by_cluster)
    optionsZ = options

    # infer the reproduction number for each of these clusters
    R_by_cluster, O_by_cluster, _ = Rt_U_O(ZData_by_cluster, lambdaU_L, lambdaU_O, options=optionsZ)

    # generate the synthetic Z data according to the R and O of their respective cluster
    ZData_by_country = []
    for i in range(nclusters):
        for _ in range(cluster_sizes[i]):
            if alpha is None:
                alpha_eff = 1
            else:
                alpha_eff = alpha*ZData_by_cluster[i][0]
            firstCases = alpha_eff * np.random.poisson(ZData_by_cluster[i][0] / alpha_eff)
            ZData, options = build.buildData_anyRO(R_by_cluster[i,:], O_by_cluster[i,:], firstCases, firstDay, alpha=alpha_eff)
            ZData_by_country.append(ZData)
    ZData_by_country = np.array(ZData_by_country)

    return ZData_by_country, ZData_by_cluster, R_by_cluster, O_by_cluster, options