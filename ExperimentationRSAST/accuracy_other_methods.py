# Import the os module
import os

path = 'c:\\Users\\Nicolas R\\random_sast'

try:
    os.chdir(path)
    print("Current working directory: {0}".format(os.getcwd()))
except FileNotFoundError:
    print("Directory: {0} does not exist".format(path))
except NotADirectoryError:
    print("{0} is not a directory".format(path))
except PermissionError:
    print("You do not have permissions to change to {0}".format(path))



#os.chdir(os.getcwd()+"/ExperimentationRSAST")
print(os.getcwd())


# %%

from sast.sast import *
from sktime.datasets import load_UCR_UEA_dataset, tsc_dataset_names
from sktime.classification.kernel_based import RocketClassifier
#from sktime.classification.shapelet_based import MrSQM
from convst.classifiers import R_DST_Ridge
from sast.utils_sast import load_dataset, format_dataset


import time
import pandas as pd


# %% [markdown]
# ### Select Datasets for hypertunning RSAST

# %% [markdown]
# It is runned RSAST in a set of UCR datasets with a predefined number of runs ("runs"). Then, it is selected a range ("range_total") between [1, 10, 30 ,50,100] for the selected dataset.

# %%

ds_sens=pd.read_excel("ExperimentationRSAST\DataSetsUCLASummary.xlsx")

ds_sens=ds_sens[ds_sens['N RUNS S17_OTHER_R10'].isna()]
#ds_sens=ds_sens[ds_sens['USED SAST']=="Y"]
ds_sens=ds_sens.Name.unique()
len(ds_sens)
'''
ds_sens = tsc_dataset_names.univariate_equal_length
#list_remove=["SmoothSubspace","Chinatown","ItalyPowerDemand","SyntheticControl","SonyAIBORobotSurface2","DistalPhalanxOutlineAgeGroup","DistalPhalanxOutlineCorrect","GunPoint","Fungi","Coffee","ShapeletSim"]
list_remove=ds_sens.Name.unique()[1:29]
# using set() to perform task
set1 = set(ds_sens)
set2 = set(list_remove)

ds_sens = list(set1 - set2)
'''






#ds_sens1 = ['SmoothSubspace', 'Car', 'ECG5000']

#ds_sens2 = ['ToeSegmentation2', 'ItalyPowerDemand','Crop']

#ds_sens=ds_sens1

ds_sens = [ 'Chinatown']


max_ds=len(ds_sens) #exploring dataset in UEA & UCR Time Series Classification Repository
print(max_ds)
print(ds_sens)

# %%
#define numbers of runs of the experiment
#define numbers of runs of the experiment
runs = 10

not_found_ds =[]



for ds in ds_sens:
    df_result = {}
    list_score = []
    list_time_fit = []
    list_time_test = []
    list_dataset = []
    list_hyperparameter = []
    list_method = []
    list_rpoint = []
    list_nb_per_class = []
    try:
        train_ds, test_ds = load_dataset("C:/Users/Nicolas R/random_sast/ExperimentationRSAST/data",ds,shuffle=True)
        X_train,y_train= format_dataset(train_ds)
        X_test,y_test= format_dataset(test_ds)
        #X_train, y_train = load_UCR_UEA_dataset(name=ds, extract_path='data', split="train", return_type="numpy2d")
        #X_test, y_test = load_UCR_UEA_dataset(name=ds, extract_path='data', split="test", return_type="numpy2d")
        X_train=np.nan_to_num(X_train)
        y_train=np.nan_to_num(y_train)
        X_test=np.nan_to_num(X_test)
        y_test=np.nan_to_num(y_test)
        print("ds="+ds)
    except:
        print("not found ds="+ds)
        not_found_ds.append(ds)
        continue

    for i in range(runs):
       
        print("ROCKET: kernels=10_000")
        start = time.time()
        rocket= RocketClassifier(num_kernels=10_000)
        rocket.fit(X_train,y_train)
        end = time.time()
        time_fit=end-start
        start = time.time()
        score=rocket.score(X_test, y_test)
        end = time.time()
        time_test=end-start

        list_score.append(score)
        list_time_fit.append(time_fit)
        list_time_test.append(time_test)
        list_dataset.append(ds)
        list_hyperparameter.append("num_kernels=10_000")

        list_method.append("Rocket")
        
        print("RDST_convst: n_shapelets=10_000")
        X_train_rdst=X_train[:, np.newaxis, :]
        y_train_rdst=np.asarray([int(x_s) for x_s in y_train])

        X_test_rdst=X_test[:, np.newaxis, :]
        y_test_rdst=np.asarray([int(x_s) for x_s in y_test])

        start = time.time()

        rdst = R_DST_Ridge(n_shapelets=10_000)
        rdst.fit(X_train_rdst, y_train_rdst)
        end = time.time()
        time_fit=end-start
        start = time.time()
        score=rdst.score(X_test_rdst, y_test_rdst)
        end = time.time()
        time_test=end-start
        
        list_score.append(score)
        list_time_fit.append(time_fit)
        list_time_test.append(time_test)
        list_dataset.append(ds)
        list_hyperparameter.append("convst: n_shapelets=10_000")

        list_method.append("RDST")
        """
        print("MrSQMC: strat=RS")


        start = time.time()

        mrsq = MrSQM() 
        mrsq.fit(X_train, y_train)
        end = time.time()
        time_fit=end-start
        start = time.time()
        score=mrsq.score(X_test, y_test)
        end = time.time()
        time_test=end-start
        
        list_score.append(score)
        list_time_fit.append(time_fit)
        list_time_test.append(time_test)
        list_dataset.append(ds)
        list_hyperparameter.append("strat=RS")

        list_method.append("MrSQM")
        
        print("RDST_aoen: n_shapelets=10_000")


        start = time.time()

        clf = RDSTClassifier()
        clf.fit(X_train_rdst, y_train_rdst)
        end = time.time()
        time_fit=end-start
        start = time.time()
        score=clf.score(X_test_rdst, y_test_rdst)
        end = time.time()
        time_test=end-start
        
        list_score.append(score)
        list_time_fit.append(time_fit)
        list_time_test.append(time_test)
        list_dataset.append(ds)
        list_hyperparameter.append("aoen: n_shapelets=10_000")

        list_method.append("RDST")
        """
        
        

    df_result['accuracy']=list_score
    df_result['time_fit']=list_time_fit
    df_result['time_test']=list_time_test
    df_result['dataset_name']=list_dataset
    df_result['hyperparameter']=list_hyperparameter

    df_result['classifier_name']=list_method
    df_result=pd.DataFrame(df_result)
    # export a overall dataset with the comparison
    df_result.to_csv("ExperimentationRSAST/results_other_methods/df_overall_comparison_results_shuffle_"+ds+".csv")
