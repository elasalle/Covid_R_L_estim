# Covid-R-estim

**Covid-R-estim** is a web repository implementing various state-of-the-art instantaneous reproduction number R(t) 
estimators, applied to Covid-19. Synthetic realistic univariate and multivariate infection counts generation as well as
tools for estimators quantitative comparison are available.
- ---

**Authors**: [Juliana Du](<https://juliana-du.github.io/>) (1), Barbara Pascal (2), Patrice Abry (1)

**Affiliations**: 
(1) CNRS, ENS de Lyon Laboratoire de physique F-69007 Lyon, France

(2) Nantes Université, École Centrale Nantes, CNRS, LS2N, UMR 6004 F-44000 Nantes, France

**Fundings**: Work supported by [ANR-23-CE48-0009 “OptiMoCSI”](<https://optimocsi.cnrs.fr>). J. Du's PhD is funded by 
CNRS 80PRIME-2021 «CoMoDécartes» project

**Created in**: June 2023, <b><i> updated in March 2024 </b> </i>

- ---
<font size="+3"> <b> NEW </b></font><font size="+2">(March 2024): Multivariate synthetic infection counts generation and
estimators comparison </font>

- ---
# Estimation of multivariate reproduction number 

The estimation codes are associated to [1], [2], [3] that present multivariate reproduction number R estimators. 

[demo_estimRealData.ipynb](demo_estimRealData.ipynb) is a Jupyter notebook that computes univariate estimations of the 
instantaneous reproduction number given daily new infection counts to be found on [Johns Hopkins University](<https://coronavirus.jhu.edu/map.html>) `JHU` 
website or [Santé Publique France](<https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/>)
`SPF`. See [README_data](<data/README_data.md>) for more details.

The different methods used are: 

* `MLE`: [Maximum Log-likelihood Estimator](<include/estim/Rt_MLE.py>)

* `Gamma`: [Bayesian Estimator](<include/estim/Rt_Gamma.py>) [1]

* `U`: [Univariate variational estimator with temporal regularization](<include/estim/Rt_Univariate.py>) [2]
* `M` : [Multivariate variational estimator with temporal and **space regularization** regularization](<include/estim/Rt_Multivariate.py>) [2]
* `U-O`: [Univariate variational estimator with temporal regularization with misreported counts `O` explicit 
modelisation](<include/estim/Rt_UnivariateOutliers.py>) [3]

[//]: # (* `M-O`: [Variational estimator with temporal and **space regularization**, with misreported counts `O` explicit )

[//]: # (modelisation]&#40;<include/estim/Rt_MultivariateOutliers.py>&#41; [3])

Code for these methods are to be found in subdirectory [include/estim/](<include/estim>).

`U`,`M` and `U-O` are variational estimators formulated as a minimization problems solved using Chambolle-Pock [4]
primal dual algorithm, customized to the objective functions involved. See implementation in 
[include/optim_tools/]((<include/optim_tools>)). 


# NEW: Multivariate synthetic infection counts generation

[demo_buildSyntheticData.ipynb](demo_buildSyntheticData.ipynb) is a Jupyter notebook that displays the generation of 
realistic univariate (for now) synthetic Covid-19 daily new infection counts, given ground truth reproduction number 
and outliers (misreported counts).


### Synthetic infection counts generation
This work is described thoroughly in [5] (in french), but also
[available on HAL](<https://hal.science/hal-04032614v2/document>).

In [data/Synthetic/Univariate](<data/Synthetic/Univariate>), you will find specific examples of ground truth resulting 
from `U-O` joint estimation strategy of reproduction number and outliers using different hyperparameters tuning more or 
less slope changes (`lambdaR`) and more or less outliers (`lambdaO`):

* [Config I](<data/Synthetic/Univariate/Config_I.mat>):     more slope changes and less zero-values in outliers 
* [Config II](<data/Synthetic/Univariate/Config_II.mat>):    less slope changes and more zero-values in outliers
* [Config III](<data/Synthetic/Univariate/Config_III.mat>):   more slope changes and more zero-values in outliers
* [Config IV](<data/Synthetic/Univariate/Config_IV.mat>):    less slope changes and less zero-values in outliers

[//]: # (Generation of synthetic infection counts files from <b> any ground truth </b> are to be found in )

[//]: # ([include/build_synth/]&#40;<include/build_synth>&#41;.)

### Spatially correlated multivariate reproduction number time series
This work is described in [6] and [available on HAL](<https://hal.science/hal-04501967>).

In [data/Synthetic/Multivariate/](<data/Synthetic/Multivariate>), you will find two examples of connectivity structure 
([`Line`](<data/Synthetic/Multivariate/Line_graph>) and [`Hub`](<data/Synthetic/Multivariate/Hub_graph>)) and for each, 
associated synthetic infection counts generated using five inter-county correlation levels : 
* [Config_delta_0](<data/Synthetic/Multivariate/Line_graph/Config_delta_0.mat>) `delta = 0` (no correlation)
* [Config_delta_I](<data/Synthetic/Multivariate/Line_graph/Config_delta_I.mat>) `delta = delta_I` (low correlation)
* [Config_delta_II](<data/Synthetic/Multivariate/Line_graph/Config_delta_II.mat>) `delta = delta_II`
* [Config_delta_III](<data/Synthetic/Multivariate/Line_graph/Config_delta_III.mat>) `delta = delta_III`
* [Config_delta_IV](<data/Synthetic/Multivariate/Line_graph/Config_delta_IV.mat>) `delta = delta_IV` (high correlation)

(example links to one connectivity structure by default).

# Comparison between R estimators on synthetic infection counts

[demo_compareEstimSynthData.ipynb](demo_compareEstimSynthData.ipynb) is a Jupyter notebook that displays:
* the comparison of the univariate methods presented earlier on generated univariate synthetic infection counts
* the comparison between univariate (independently) and multivariate strategies on multivariate synthetic infection counts

MSE, SNR and Jaccard index are used for comparison are described in 
[include/comparison_tools](<include/comparison_tools/>).

- ---
Notes: 
* all the display functions are to be found in [display/](<display>).
* previous version is available in [archives/](<archives/Covid-R-estim-GRETSI23.zip>) in `.zip` format.

# References 
[1] A. Cori, N. Ferguson, C. Fraser, and S. Cauchemez, “A new framework and software to estimate time-varying 
reproduction numbers during epidemics,” Am. J. Epidemiol., vol. 178, no. 9, pp. 1505–1512, 2013.

[2] P. Abry, N. Pustelnik, S. Roux, P. Jensen, P. Flandrin, R. Gribonval, C.-G. Lucas,  ́E. Guichard, P. Borgnat, and
N. Garnier, “Spatial and temporal regularization to estimate COVID-19 reproduction number R(t): Promoting piecewise 
smoothness via convex optimization,” PLoS One, vol. 15, no. 8, p. e0237901, 2020. [⟨hal-02921836⟩](<https://hal.science/hal-02921836/>)

[3] B. Pascal, P. Abry, N. Pustelnik, S. Roux, R. Gribonval, and P. Flandrin, “Nonsmooth convex optimization to estimate
the Covid-19 reproduction number space-time evolution with robustness against low quality data,” IEEE Trans. Signal 
Process., vol. 70, pp. 2859–2868, 2022. [available on arXiv](<https://arxiv.org/abs/2109.09595>)

[4] A. Chambolle and T. Pock, “A first-order primal-dual algorithm for convex problems with applications to imaging,” 
J. Math. Imaging Vis., vol. 40, no. 1, pp. 120–145, 2011.

[5] J. Du, B. Pascal, and P. Abry, “Compared performance of Covid19 reproduction number estimators based on realistic 
synthetic data,” in GRETSI’23 XXIX`eme Colloque Francophone de Traitement du Signal et des Images, Grenoble, France, 
Aug. 28 - Sept. 1 2023. [⟨hal-04032614⟩](<https://hal.science/hal-04032614v2/document>)


[6] J. Du, B. Pascal, P. Abry. "Synthetic Spatiotemporal Covid19 Infection Counts to Assess Graph-Regularized Estimation
of Multivariate Reproduction Numbers". 2024. [⟨hal-04501967⟩](<https://hal.science/hal-04501967>)