# Covid-R-estim data

In this file, you will find details about infection counts data available in this directory.
- ---

## Real-world infection counts

* [counties_sharing_borders.mat](<Real-world/counties_sharing_borders.mat>) contains the matrix Id - A where A is the 
adjacency matrix of the graph representing sharing borders french counties (*départements*).


* [JHU-worldwide-covid19-daily-new-infections.csv](<Real-world/JHU-worldwide-covid19-daily-new-infections.csv>) 
*downloaded on 19/03/2024 (9h34)* from [Johns Hopkins University website](<https://coronavirus.jhu.edu/map.html>)
contains daily new infection counts from 100+ countries. It has stopped collecting new Covid-19 data as of 2023-03-10.


* [SiDEP-France-by-day-2023-06-30-16h26.csv](<Real-world/SiDEP-France-by-day-2023-06-30-16h26.csv>) 
*downloaded on 19/03/2024 (9h34)* from [Santé Publique France website](<https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/>)
contains daily new infection counts for France. It has stopped collecting new Covid-19 data as of 2023-04-01.


* [SiDEP-France-by-day-by-dep-2023-06-30-16h26.csv](<Real-world/SiDEP-France-by-day-by-dep-2023-06-30-16h26.csv>) 
*downloaded on 19/03/2024 (9h34)* from [Santé Publique France website](<https://www.data.gouv.fr/fr/datasets/donnees-de-laboratoires-pour-le-depistage-a-compter-du-18-05-2022-si-dep/>)
contains daily new infection counts for France by french counties *département*. It has stopped collecting new Covid-19 
data as of 2023-04-01.


* [SiDEP-France-hosp-2023-03-31-18h01.csv](<Real-world/SiDEP-France-hosp-2023-03-31-18h01.csv>) 
*downloaded on 19/03/2024 (9h34)* from [Santé Publique France website](<https://www.data.gouv.fr/fr/datasets/donnees-hospitalieres-relatives-a-lepidemie-de-covid-19/>)
contains daily new hospitalization counts for France. It has stopped collecting new Covid-19 data as of 2023-04-01.


## Synthetic infection counts 

### Univariate
In [data/Synthetic/Univariate](<Synthetic/Univariate>), you will find specific examples of ground truth:

* [Config I](<Synthetic/Univariate/Config_I.mat>):     more slope changes and less zero-values in outliers 
* [Config II](<Synthetic/Univariate/Config_II.mat>):    less slope changes and more zero-values in outliers
* [Config III](<Synthetic/Univariate/Config_III.mat>):   more slope changes and more zero-values in outliers
* [Config IV](<Synthetic/Univariate/Config_IV.mat>):    less slope changes and less zero-values in outliers

### Multivariate
In [data/Synthetic/Multivariate/](<Synthetic/Multivariate>), you will find two examples of connectivity structure 
[`Line_graph`](<Synthetic/Multivariate/Line_graph>) and [`Hub_graph`](<Synthetic/Multivariate/Hub_graph>).

In each directory, associated synthetic infection counts generated using five inter-county correlation levels : 
* [Config_delta_0](<Synthetic/Multivariate/Line_graph/Config_delta_0.mat>) `delta = 0` (no correlation)
* [Config_delta_I](<Synthetic/Multivariate/Line_graph/Config_delta_I.mat>) `delta = delta_I` (low correlation)
* [Config_delta_II](<Synthetic/Multivariate/Line_graph/Config_delta_II.mat>) `delta = delta_II`
* [Config_delta_III](<Synthetic/Multivariate/Line_graph/Config_delta_III.mat>) `delta = delta_III`
* [Config_delta_IV](<Synthetic/Multivariate/Line_graph/Config_delta_IV.mat>) `delta = delta_IV` (high correlation)
