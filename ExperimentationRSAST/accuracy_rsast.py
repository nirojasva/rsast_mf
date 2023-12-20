# Import the os module
import os
import sys
path = '/home/nicolas/mf_rsast'


try:
    os.chdir(path)
    print("Current working directory: {0}".format(os.getcwd()))
except FileNotFoundError:
    print("Directory: {0} does not exist".format(path))
except NotADirectoryError:
    print("{0} is not a directory".format(path))
except PermissionError:
    print("You do not have permissions to change to {0}".format(path))

sys.path.append(path+'/sast')
#os.chdir(os.getcwd()+"/ExperimentationRSAST")
print(os.getcwd())


# %%
from sast import *
from sktime.datasets import load_UCR_UEA_dataset, tsc_dataset_names
from sktime.classification.kernel_based import RocketClassifier
import time
import pandas as pd
from utils_sast import load_dataset, format_dataset


# %% [markdown]
# ### Select Datasets for hypertunning RSAST

# %% [markdown]
# It is runned RSAST in a set of UCR datasets with a predefined number of runs ("runs"). Then, it is selected a range ("range_total") between [1, 10, 30 ,50,100] for the selected dataset.

# %%

ds_sens=pd.read_excel("ExperimentationRSAST/DataSetsUCLASummary.xlsx")

#ds_sens=ds_sens[ds_sens['N RUNS S17_RSAST_R10'].isna()]
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

ds_sens=["Chinatown"]

#ds_sens = [ 'WormsTwoClass','Mallat','StarLightCurves','UWaveGestureLibraryAll','Phoneme','NonInvasiveFetalECGThorax1','NonInvasiveFetalECGThorax2']


max_ds=len(ds_sens) #exploring dataset in UEA & UCR Time Series Classification Repository
print(max_ds)
print(ds_sens)

# %%
#define numbers of runs of the experiment
runs = 10

#define range for number of random points 
range_rpoint = [10, 30]

#define range for number of intances per class
range_nb_inst_per_class=[1, 10]

#define range for quantiles p and q
range_p=[0, 0.1,0.2,0.3]
range_q=[0, 0.1,0.2,0.3]

#define lenght method
len_methods = ["both", "PACF", "ACF"]


not_found_ds =[]

for ds in ds_sens:

    try:
        
        train_ds, test_ds = load_dataset("ExperimentationRSAST/data",ds,shuffle=False)
        
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
        df_result = {}
        list_score = []
        list_overall_time = []
        list_cweight_time = []
        list_fsubsequence_time = []
        list_tdataset_time = []
        list_tclassifier_time = []
        list_dataset = []
        list_hyperparameter = []
        list_method = []
        list_range_p = []
        list_range_q = []
        list_len_method = []
        for len_m in len_methods:
            for p in range_p:
                for q in range_q:
                    if q>=p:
                        if len_m=="both":
                            len_m_corrected="ACF&PACF" 
                        else:
                            len_m_corrected=len_m
                        print("Rsastmf"+len_m_corrected+": p="+str(p)+" q="+str(q))
                        start = time.time()
                        random_state = None
                        rsastmf_ridge = RSASTMF(n_random_points=10,nb_inst_per_class=10, len_method=len_m, q_max=q, q_min=p)
                        rsastmf_ridge.fit(X_train, y_train)
                        end = time.time()
                        
                        
                        list_score.append(rsastmf_ridge.score(X_test,y_test))

                        list_overall_time.append(end-start)
                        list_cweight_time.append(rsastmf_ridge.time_calculating_weights)
                        list_fsubsequence_time.append(rsastmf_ridge.time_creating_subsequences)
                        list_tdataset_time.append(rsastmf_ridge.transform_dataset)
                        list_tclassifier_time.append(rsastmf_ridge.time_classifier)

                        list_dataset.append(ds)
                        list_hyperparameter.append(len_m_corrected+": p="+str(p)+" q="+str(q))
                        list_range_p.append(str(p))
                        list_range_q.append(str(q))
                        list_method.append("Rsastmf")
                        list_len_method.append(len_m_corrected)

                        print("DictRsast"+len_m_corrected+": p="+str(p)+" q="+str(q))
                        start = time.time()
                        random_state = None
                        dictrsast_ridge = DICTRSAST(n_random_points=10,nb_inst_per_class=10, len_method=len_m, q_max=q, q_min=p)
                        dictrsast_ridge.fit(X_train, y_train)
                        end = time.time()
                        
                        
                        list_score.append(dictrsast_ridge.score(X_test,y_test))

                        list_overall_time.append(end-start)
                        list_cweight_time.append(dictrsast_ridge.time_calculating_weights)
                        list_fsubsequence_time.append(dictrsast_ridge.time_creating_subsequences)
                        list_tdataset_time.append(dictrsast_ridge.transform_dataset)
                        list_tclassifier_time.append(dictrsast_ridge.time_classifier)

                        list_dataset.append(ds)
                        list_hyperparameter.append(len_m_corrected+": p="+str(p)+" q="+str(q))
                        list_range_p.append(str(p))
                        list_range_q.append(str(q))
                        list_method.append("DictRsast")
                        list_len_method.append(len_m_corrected)
                    
        df_result['accuracy']=list_score
        df_result['time']=list_overall_time
        df_result['cweights_time']=list_cweight_time
        df_result['fsubsequence_time']=list_fsubsequence_time
        df_result['tdataset_time']=list_tdataset_time
        df_result['tclassifier_time']=list_tclassifier_time
        df_result['dataset_name']=list_dataset
        df_result['classifier_name']=list_hyperparameter
        df_result['range_p']=list_range_p
        df_result['range_q']=list_range_q
        df_result['method']=list_method
        df_result['len_method']=list_len_method
        df_result=pd.DataFrame(df_result)
        # export a overall dataset with results
        df_result.to_csv("ExperimentationRSAST/results_rsast/df_all_overall_tunning_mf_shuffle_"+str(ds)+str(i+1)+"_norepTSRP.csv") 


