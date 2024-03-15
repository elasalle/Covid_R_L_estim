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

The estimation codes are associated to the paper written by B. Pascal, P. Abry, N. Pustelnik, S. Roux, R. Gribonval, 
and P. Flandrin, “Nonsmooth convex optimization to estimate the Covid-19 reproduction number space-time evolution with 
robustness against low quality data,” IEEE Trans. Signal Process., vol. 70, pp. 2859–2868, 2022.
    
In this repository, you'll find 

    demo_estimRealData.ipynb


which is a Jupyter notebook that computes estimations of time series reproduction number
given daily new cases to be found in <a href="https://coronavirus.jhu.edu/map.html">Johns Hopkins University </a> 'JHU' 
or <a href="https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/">
Santé Publique France </a>
'SPF'. See <i> include/load_data </i> for more details on the data.

The different methods used are: 

* (MLE) : Maximum Log-likelihood Estimator

* (C) : Bayesian Estimator [1]

* (U), (U-O) : Univariate variational estimator with temporal regularization, without or with (O) misreported counts explicit modelisation [2]

* (M), (M-O) : Multivariate variational estimator with temporal and space regularization, without or with (O) misreported counts explicit modelisation [3]

[//]: # (* &#40;PLG&#41; : Penalized Loglikelihood &#40;time and space regularized&#41;)

[//]: # ()
[//]: # (* &#40;U-O&#41; : Joint)

Code for these methods are to be found in subdirectory <i> include/estim/ </i>.

(U) and (M) are the results of solving optimization schemes, which tools are described in <i> include/optim_tools/ </i>  


# NEW: Multivariate synthetic infection counts generation

### Synthetic infection counts generation
This work is associated to the paper written by J. Du, B. Pascal, and P. Abry, 
“Compared performance of Covid19 reproduction number estimators based on realistic synthetic data,” 
in GRETSI’23 XXIX`eme Colloque Francophone de Traitement du Signal et des Images, Grenoble, France, 
Aug.28 - Sept. 1 2023, also <a href="https://hal.science/hal-04032614v2/document"> available on HAL </a>.


    demo_buildSyntheticData.ipynb

is a Jupyter notebook that displays how to generate realistic synthetic data of daily new cases for Covid-19 pandemic,
given ground truth reproduction number and outliers.

Specific examples of ground truth are to be found in <i> data/ </i>. 

They are resulting from Joint (J) estimations of reproduction number and outliers using different hyperparameters 
tuning more or less slope changes (lambdaR) and more or less outliers (lambdaO):

* Config I:     more slope changes and less zero-values in outliers 
* Config II:    less slope changes and more zero-values in outliers
* Config III:   more slope changes and more zero-values in outliers
* Config IV:    less slope changes and less zero-values in outliers

Generation of synthetic infection counts files from <b> any ground truth </b> are to be found in <i> include/build_synth/ </i>.

### Spatially correlated multivariate reproduction number time series
This work is associated to the paper written by J. Du, B. Pascal, and P. Abry, "Synthetic Spatiotemporal Covid19 
Infection Counts to Assess Graph-Regularized Estimation of Multivariate Reproduction Numbers" 
[available on HAL](<https://hal.science/hal-04501967>).



# Comparison between R estimators on synthetic infection counts

    demo_compareEstimSynthData.ipynb

is a Jupyter notebook that displays comparison of the 4 methods presented earlier on generated synthetic data.

The quantities SNR and Jaccard index used for comparison are described in <i> include/comparison_tools/ </i>.

Note : all the display functions are to be found in <i> display/ </i>.
