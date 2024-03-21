# Covid-R-estim

**Covid-R-estim** is a web repository implementing various state-of-the-art instantaneous reproduction number R(t) 
estimators, applied to Covid-19. Synthetic realistic univariate and multivariate infection counts generation as well as
tools for estimators quantitative comparison are available.
- ---

**Author**: [Juliana Du](<https://juliana-du.github.io/>)

**Affiliation**: ENSL, CNRS, Laboratoire de physique

**Funding**: Work supported by ANR-23-CE48-0009 “OptiMoCSI”. J. Du's PhD is funded by CNRS 80PRIME-2021 «CoMoDécartes» 
project

**Created in**: June 2023, <b><i> updated in March 2024 </b> </i>

- ---
<font size="+3"> <b> NEW</b></font><font size="+2"> (March 2024): Multivariate synthetic infection counts generation and
estimators comparison available. </font>

- ---
# Estimation of multivariate reproduction number 

The estimation codes are associated to [1], [2], [3] that present multivariate reproduction number R estimators. 

[demo_estimRealData.ipynb](demo_estimRealData.ipynb) is a Jupyter notebook that computes univariate estimations of the instantaneous reproduction number given 
daily new infection counts to be found on [Johns Hopkins University](<https://coronavirus.jhu.edu/map.html>) `JHU` 
website or [Santé Publique France](<https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/>)
`SPF`. See [include/load_data/load_counts.py](include/load_data/load_counts.py) for more details.

The different methods used are: 

* `MLE`: Maximum Log-likelihood Estimator

* `Gamma`: Bayesian Estimator [1]

* `U`: Univariate variational estimator with temporal regularization [2]
* `M` : Multivariate variational estimator with temporal and **space regularization** regularization [2]
* `U-O`: Univariate variational estimator with temporal regularization with misreported counts `O` explicit 
modelisation [3]
* `M-O`: Variational estimator with temporal and **space regularization**, with misreported counts `O` explicit 
modelisation [3]

Code for these methods are to be found in subdirectory `include/estim/`.

`U`,`M`, `U-O`and `M-O` are variational estimators formulated as a minimization problems solved using Chambolle-Pock [4]
primal dual algorithm, customized to the objective functions involved. See implementation in `include/optim_tools/`. 


# NEW: Multivariate synthetic infection counts generation

### Synthetic infection counts generation
This work is described thoroughly in [5] (in french), but also
[available on HAL](<https://hal.science/hal-04032614v2/document>).

[demo_buildSyntheticData.ipynb](demo_buildSyntheticData.ipynb) is a Jupyter notebook that displays the generation of realistic univariate (for now) synthetic Covid-19 daily new 
infection counts, given ground truth reproduction number and outliers (misreported counts).

Specific examples of ground truth are to be found in `data/Synthetic/Univariate`. 

They are resulting from `U-O` joint estimation strategy of reproduction number and outliers using different 
hyperparameters tuning more or less slope changes (`lambdaR`) and more or less outliers (`lambdaO`):

* Config I:     more slope changes and less zero-values in outliers 
* Config II:    less slope changes and more zero-values in outliers
* Config III:   more slope changes and more zero-values in outliers
* Config IV:    less slope changes and less zero-values in outliers

Generation of synthetic infection counts files from <b> any ground truth </b> are to be found in 
`include/build_synth/`.

### Spatially correlated multivariate reproduction number time series
This work is described in [6] and [available on HAL](<https://hal.science/hal-04501967>).

In `data/Synthetic/Multivariate/` you will find two examples of connectivity structure (`Line` and `Hub`) and for each, 
associated synthetic infection counts generated using five inter-county correlation levels : `delta = 0` (no correlation), 
`delta_I` (low correlation), `delta_II`,`delta_III`, `delta_IV` (high correlation).

# Comparison between R estimators on synthetic infection counts

[demo_compareEstimSynthData.ipynb](demo_compareEstimSynthData.ipynb) is a Jupyter notebook that displays comparison of the 4 univariate methods presented earlier on generated univariate 
synthetic infection counts. Soon to be available : multivariate comparison.

SNR, MSE and Jaccard index are used for comparison are described in `include/comparison_tools/`.

Note : all the display functions are to be found in `display/`.

# References 
[1] A. Cori, N. Ferguson, C. Fraser, and S. Cauchemez, “A new framework and software to estimate time-varying 
reproduction numbers during epidemics,” Am. J. Epidemiol., vol. 178, no. 9, pp. 1505–1512, 2013.

[2] P. Abry, N. Pustelnik, S. Roux, P. Jensen, P. Flandrin, R. Gribonval, C.-G. Lucas,  ́E. Guichard, P. Borgnat, and
N. Garnier, “Spatial and temporal regularization to estimate COVID-19 reproduction number R(t): Promoting piecewise 
smoothness via convex optimization,” PLoS One, vol. 15, no. 8, p. e0237901, 2020.

[3] B. Pascal, P. Abry, N. Pustelnik, S. Roux, R. Gribonval, and P. Flandrin, “Nonsmooth convex optimization to estimate
the Covid-19 reproduction number space-time evolution with robustness against low quality data,” IEEE Trans. Signal 
Process., vol. 70, pp. 2859–2868, 2022.

[4] A. Chambolle and T. Pock, “A first-order primal-dual algorithm for convex problems with applications to imaging,” 
J. Math. Imaging Vis., vol. 40, no. 1, pp. 120–145, 2011.

[5] J. Du, B. Pascal, and P. Abry, “Compared performance of Covid19 reproduction number estimators based on realistic 
synthetic data,” in GRETSI’23 XXIX`eme Colloque Francophone de Traitement du Signal et des Images, Grenoble, France, 
Aug. 28 - Sept. 1 2023. [⟨hal-04032614⟩](<https://hal.science/hal-04032614v2/document>)


[6] J. Du, B. Pascal, P. Abry. "Synthetic Spatiotemporal Covid19 Infection Counts to Assess Graph-Regularized Estimation
of Multivariate Reproduction Numbers". 2024. [⟨hal-04501967⟩](<https://hal.science/hal-04501967>)