# Covid-R-estim


**Covid-R-estim** is a web repository that gives tools for instaneous reproduction number R(t) estimation, applied to Covid-19.
Synthetic data generation and more tools for estimators quantitative comparison are available.
- ---

**Author**: [Juliana Du](<https://juliana-du.github.io/>)

**Affiliation**: ENSL, CNRS, Laboratoire de physique

**Funding**: CNRS 80PRIME-2021 «CoMoDécartes» project

**Created in**: June 2023

- ---

# Estimation of reproduction number 

The estimation codes are associated to the paper "Nonsmooth convex optimization to estimate the Covid-19 reproduction
number space-time evolution with robustness against low quality data." written by B. Pascal, P. Abry, N. Pustelnik, 
S.G. Roux, R. Gribonval and P. Flandrin, Trans. Signal Process. 2022
    
In this repository, you'll find 

    demo_estimRealData.ipynb


which is a Jupyter notebook that computes estimations of time series reproduction number
given daily new cases to be found in Johns Hopkins University (https://coronavirus.jhu.edu/map.html) 'JHU' or Santé 
Publique France (https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/) 
'SPF'. See <i> include/load_data </i> for more details on the data.

The different methods used are: 

* (MLE) : Maximum Loglikelihood Estimator

* (C) : Cori's method

* (PL) : Penalized Loglikelihood

* (J) : Joint

Code for these methods are to be found in subdirectory <i> include/estim/ </i>.

(PL) and (J) are the results of solving optimization schemes, which tools are described in <i> include/optim_tools/ </i>  

# Synthetic data generation
The data generation codes are associated to the paper 
"Compared performance of Covid19 reproduction number estimators based on realistic synthetic data" written by J.Du, 
B. Pascal and P. Abry, GRETSI 2023 proceedings.


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

Generation of synthetic data files from <b> any ground truth </b> are to be found in <i> include/build_synth/ </i>.


# Comparison between estimated R on synthetic data

    demo_compareEstymSynthData.ipynb

is a Jupyter notebook that displays comparison of the 4 methods presented earlier on generated synthetic data.

The quantities SNR and Jaccard index used for comparison are described in <i> include/comparison_tools/ </i>.

Note : all the display functions are to be found in <i> display/ </i>.
