# RSAST: Random Scalable and Accurate Subsequence Transform for Time Series Classification

Random SAST (RSAST) is a method based on STC and SAST, that generates shapelets randomly, guided by certain statistical
criteria, reducing the search space of shapelets.


## Results RSAST

- [Results Default Split](./ExperimentationRSAST/results_default_split.csv)

- [Results 10 Resamplings](./ExperimentationRSAST/results_10resampling.csv)

- [Results Comparison RSAST](./ExperimentationRSAST/results_comparison_rsast.csv)

- [Execution time regarding the number of series](./ExperimentationRSAST/results_comparison_accuracy/df_overall_comparison_scalability_number_of_seriesLR.csv)

- [Execution time regarding series length](./ExperimentationRSAST/results_comparison_accuracy/df_overall_comparison_scalability_series_length.csv)



## RSAST, SAST and STC

### Critical difference diagram

![](./ExperimentationRSAST/images_cd_diagram/comparison_rsast_sast_st.png)

### Pairwise accuracy comparison

| ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsSAST.png) | ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsSTC.png) |
| -------------------------------------------------- | ---------------------------------------------------- |


## Shapelet Approaches

### Critical difference diagram

![](./ExperimentationRSAST/images_cd_diagram/comparison_shapelet_method.png)

### Pairwise accuracy comparison

| ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsFS.png) | ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsLS.png) | ![](./ExperimentationRSAST/images_one_vs_one_comparison/RSASTvsRDST.png) |
| ----------------------------------------- | ------------------------------------------- | ------------------------------------------- |

## Alternative Length Methods

In order to explore another alternatives for the default length method of the shapelets (ACF&PACF) some supplementary length methods are examined: Max PACF and None.

### Critical difference diagram per length method

- The default behaviour implies chose all significant values from ACF and PACF tests.
![](./ExperimentationRSAST/images_cd_diagram/cd-diagram_ACF&PACF.png) 

- Max ACF, makes reference to the generation of subsequences considering solely the highest significant value from the Partial Autocorrelation Function (PACF).
![](./ExperimentationRSAST/images_cd_diagram/cd-diagram_Max_PACF.png) 

- "None" variation, involves generating subsequences with a single random length chosen from the range between 3 and the size of the time series for each randomly selected instance.
![](./ExperimentationRSAST/images_cd_diagram/cd-diagram_None.png) 


### Critical difference diagram: Best Performance Comparison

![](./ExperimentationRSAST/images_cd_diagram/cd-diagram_best_com.png)

## Scalability

- Regarding the length of time series:
![](./ExperimentationRSAST/images_scalability/scalability_length.png)

- Regarding the number of time series in the dataset:
![](./ExperimentationRSAST/images_scalability/scalability_ns.png)


## Usage

```python

import os, numpy as np
from utils_sast import load_dataset, format_dataset
from sast import RSAST
from sklearn.linear_model import RidgeClassifierCV

ds='Coffee' # Chosing a dataset 
path=os.getcwd()+"/data"

ds_train , ds_test = load_dataset(ds_folder=path,ds_name=ds,shuffle=False)
X_test, y_test = format_dataset(ds_test)
X_train, y_train = format_dataset(ds_train)
clf = RidgeClassifierCV(alphas=np.logspace(-3, 3, 10))
rsast_ridge = RSAST(n_random_points=10, nb_inst_per_class=10, len_method="both", classifier=clf)
rsast_ridge.fit(X_train, y_train)
print('rsast score :', rsast_ridge.score(X_test, y_test))

```
