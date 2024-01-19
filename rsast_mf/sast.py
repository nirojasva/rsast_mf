# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np

from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils.validation import check_array, check_X_y, check_is_fitted

from sklearn.ensemble import RandomForestClassifier, VotingClassifier

from sklearn.linear_model import RidgeClassifierCV, LogisticRegressionCV, LogisticRegression, RidgeClassifier


from sklearn.linear_model._base import LinearClassifierMixin
from sklearn.pipeline import Pipeline

#from sktime.utils.data_processing import from_2d_array_to_nested
#from sktime.transformations.panel.rocket import Rocket

from numba import njit, prange

from mass_ts import *

import pandas as pd

from scipy.stats import f_oneway, DegenerateDataWarning, ConstantInputWarning
from statsmodels.tsa.stattools import acf, pacf
#from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

import time

import os
from operator import itemgetter



from utils_sast import from_2d_array_to_nested, znormalize_array, load_dataset, format_dataset, plot_most_important_features, plot_most_important_feature_on_ts, plot_most_important_feature_sast_on_ts
from aeon.classification.shapelet_based import RDSTClassifier
#from sktime.datasets import load_UCR_UEA_dataset

import random
import math



@njit(fastmath=False)
def apply_kernel(ts, arr):
    d_best = np.inf  # sdist
    m = ts.shape[0]
    kernel=arr
    #kernel = arr[~np.isnan(arr)]  # ignore nan
    kernel = arr[~np.isinf(arr)]  # ignore inf

    # profile = mass2(ts, kernel)
    # d_best = np.min(profile)

    l = kernel.shape[0]
    for i in range(m - l + 1):
        
        d = np.nansum((znormalize_array(ts[i:i+l]) - kernel)**2)
        if d < d_best:
            d_best = d

    return d_best


def get_lambda_rdst(ts, arr, q_max, q_min):

    m = ts.shape[0]
    kernel=arr
    #kernel = arr[~np.isnan(arr)]  # ignore nan
    kernel = arr[~np.isinf(arr)]  # ignore inf
    d_vector = []


    l = kernel.shape[0]

    for i in range(m - l + 1):
        
        d = np.nansum((znormalize_array(ts[i:i+l]) - kernel)**2)
        d_vector.append(d)
    
    quantiles = np.quantile(d_vector, [q_min, q_max])
    
    
    random_value = np.random.uniform(quantiles[0], quantiles[1])
    

    return random_value

@njit(fastmath=False)
def apply_kernel_mf(ts, arr, lm):
    d_best = np.inf  # sdist
    p_best = 0  
    m = ts.shape[0]
    kernel=arr
    
    #kernel = arr[~np.isnan(arr)]  # ignore nan
    kernel = arr[~np.isinf(arr)]  # ignore inf

    d_vector = []
    occ=0
    # profile = mass2(ts, kernel)
    # d_best = np.min(profile)

    l = kernel.shape[0]

    for i in range(m - l + 1):
        
        d = np.nansum((znormalize_array(ts[i:i+l]) - kernel)**2)
        d_vector.append(d)
        if d < d_best:
            d_best = d
            p_best = i

    for dist in d_vector:
        
        #if dist <= d_best*(1+q_max):
        if dist < lm:    
            occ=occ+1
    #print("d_vector")
    #print(d_vector)
    #print("d_best", "p_best" , "occ")
    #print(d_best, p_best , occ)
    return d_best, p_best , occ

@njit(fastmath=False)
def apply_kernel_dict(ts, arr, q_max, q_min):
    d_best = np.inf  # sdist
    p_best = 0  
    m = ts.shape[0]
    kernel=arr
    
    #kernel = arr[~np.isnan(arr)]  # ignore nan
    kernel = arr[~np.isinf(arr)]  # ignore inf

    d_vector = []
    occ=0
    # profile = mass2(ts, kernel)
    # d_best = np.min(profile)

    l = kernel.shape[0]

    for i in range(m - l + 1):
        
        d = np.nansum((znormalize_array(ts[i:i+l]) - kernel)**2)
        d_vector.append(d)
        if d < d_best:
            d_best = d
            p_best = i
    quantiles = np.quantile(d_vector, [q_min, q_max])
    for dist in d_vector:
        
        #if dist <= d_best*(1+q_max):
        if dist >= quantiles[0] and dist <= quantiles[1]:    
            occ=occ+1
    #print("d_vector")
    #print(d_vector)
    #print("d_best", "p_best" , "occ")
    #print(d_best, p_best , occ)
    return quantiles[0], quantiles[1] , occ

@njit(parallel=True, fastmath=True)
def apply_kernels(X, kernels):
    nbk = len(kernels)
    out = np.zeros((X.shape[0], nbk), dtype=np.float32)
    for i in prange(nbk):
        k = kernels[i]
        for t in range(X.shape[0]):
            ts = X[t]
            out[t][i] = apply_kernel(ts, k)
    return out

@njit(parallel=True, fastmath=True)
def apply_kernels_dict(X, kernels, q, q_min):
    nbk = len(kernels)
    out = np.zeros((X.shape[0], nbk*3), dtype=np.float32)
    for i in prange(nbk):
        k = kernels[i]
        for t in range(X.shape[0]):
            ts = X[t]
            out[t][i*3], out[t][i*3+1], out[t][i*3+2] = apply_kernel_dict(ts, k, q, q_min)
    return out

@njit(parallel=True, fastmath=True)
def apply_kernels_mf(X, kernels, lms):
    #print("Transforming Dataset with shape:"+str(X.shape))
    #print("Total kernel: "+str(len(kernels)))
    nbk = len(kernels)
    out = np.zeros((X.shape[0], nbk*3), dtype=np.float32)
    for i in prange(nbk):
        k = kernels[i]
        l = lms[i]
        for t in range(X.shape[0]):
            #print("TS: "+str(t)+" kernel: "+str(i))
            ts = X[t]
            #out[t][i*3], out[t][i*3+1], out[t][i*3+2] = apply_kernel_mf(ts, k, q)
            out[t][i*3], out[t][i*3+1], out[t][i*3+2] = apply_kernel_mf(ts, k, l)
            
    
    
    return out

class SAST(BaseEstimator, ClassifierMixin):

    def __init__(self, cand_length_list, shp_step=1, nb_inst_per_class=1, random_state=None, classifier=None):
        super(SAST, self).__init__()
        self.cand_length_list = cand_length_list
        self.shp_step = shp_step
        self.nb_inst_per_class = nb_inst_per_class
        self.kernels_ = None
        self.kernel_orig_ = None  # not z-normalized kernels
        self.kernels_generators_ = {}
        self.random_state = np.random.RandomState(random_state) if not isinstance(
            random_state, np.random.RandomState) else random_state

        self.classifier = classifier

    def get_params(self, deep=True):
        return {
            'cand_length_list': self.cand_length_list,
            'shp_step': self.shp_step,
            'nb_inst_per_class': self.nb_inst_per_class,
            'classifier': self.classifier
        }

    def init_sast(self, X, y):

        self.cand_length_list = np.array(sorted(self.cand_length_list))

        assert self.cand_length_list.ndim == 1, 'Invalid shapelet length list: required list or tuple, or a 1d numpy array'

        if self.classifier is None:
            self.classifier = RandomForestClassifier(
                min_impurity_decrease=0.05, max_features=None)

        classes = np.unique(y)
        self.num_classes = classes.shape[0]

        candidates_ts = []
        for c in classes:
            X_c = X[y == c]

            # convert to int because if self.nb_inst_per_class is float, the result of np.min() will be float
            cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int)
            choosen = self.random_state.permutation(X_c.shape[0])[:cnt]
            candidates_ts.append(X_c[choosen])
            self.kernels_generators_[c] = X_c[choosen]

        candidates_ts = np.concatenate(candidates_ts, axis=0)

        self.cand_length_list = self.cand_length_list[self.cand_length_list <= X.shape[1]]

        max_shp_length = max(self.cand_length_list)

        n, m = candidates_ts.shape

        n_kernels = n * np.sum([m - l + 1 for l in self.cand_length_list])

        self.kernels_ = np.full(
            (n_kernels, max_shp_length), dtype=np.float32, fill_value=np.nan)
        self.kernel_orig_ = []

        k = 0

        for shp_length in self.cand_length_list:
            for i in range(candidates_ts.shape[0]):
                for j in range(0, candidates_ts.shape[1] - shp_length + 1, self.shp_step):
                    end = j + shp_length
                    can = np.squeeze(candidates_ts[i][j: end])
                    self.kernel_orig_.append(can)
                    self.kernels_[k, :shp_length] = znormalize_array(can)

                    k += 1
        
    def fit(self, X, y):

        X, y = check_X_y(X, y)  # check the shape of the data

        # randomly choose reference time series and generate kernels
        self.init_sast(X, y)

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        self.classifier.fit(X_transformed, y)  # fit the classifier

        return self

    def predict(self, X):

        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        if isinstance(self.classifier, LinearClassifierMixin):
            return self.classifier._predict_proba_lr(X_transformed)
        return self.classifier.predict_proba(X_transformed)


class SASTEnsemble(BaseEstimator, ClassifierMixin):

    def __init__(self, cand_length_list, shp_step=1, nb_inst_per_class=1, random_state=None, classifier=None, weights=None, n_jobs=None):
        super(SASTEnsemble, self).__init__()
        self.cand_length_list = cand_length_list
        self.shp_step = shp_step
        self.nb_inst_per_class = nb_inst_per_class
        self.classifier = classifier
        self.random_state = random_state
        self.n_jobs = n_jobs

        self.saste = None

        self.weights = weights

        assert isinstance(self.classifier, BaseEstimator)

        self.init_ensemble()

    def init_ensemble(self):
        estimators = []
        for i, candidate_lengths in enumerate(self.cand_length_list):
            clf = clone(self.classifier)
            sast = SAST(cand_length_list=candidate_lengths,
                        nb_inst_per_class=self.nb_inst_per_class,
                        random_state=self.random_state,
                        shp_step=self.shp_step,
                        classifier=clf)
            estimators.append((f'sast{i}', sast))

        self.saste = VotingClassifier(
            estimators=estimators, voting='soft', n_jobs=self.n_jobs, weights=self.weights)

    def fit(self, X, y):
        self.saste.fit(X, y)
        return self

    def predict(self, X):
        return self.saste.predict(X)

    def predict_proba(self, X):
        return self.saste.predict_proba(X)



class RSAST(BaseEstimator, ClassifierMixin):

    def __init__(self,n_random_points=10, nb_inst_per_class=10, len_method="both", random_state=None, classifier=None, sel_inst_wrepl=False,sel_randp_wrepl=False, half_instance=False, half_len=False,n_shapelet_samples=None ):
        super(RSAST, self).__init__()
        self.n_random_points = n_random_points
        self.nb_inst_per_class = nb_inst_per_class
        self.len_method = len_method
        self.random_state = np.random.RandomState(random_state) if not isinstance(
            random_state, np.random.RandomState) else random_state
        self.classifier = classifier
        self.cand_length_list = None
        self.kernels_ = None
        self.kernel_orig_ = None  # not z-normalized kernels
        self.kernel_permutated_ = None
        self.kernels_generators_ = None
        self.class_generators_ = None
        self.sel_inst_wrepl=sel_inst_wrepl
        self.sel_randp_wrepl=sel_randp_wrepl
        self.half_instance=half_instance
        self.half_len=half_len
        self.time_calculating_weights = None
        self.time_creating_subsequences = None
        self.time_transform_dataset = None
        self.time_classifier = None
        self.n_shapelet_samples =n_shapelet_samples

    def get_params(self, deep=True):
        return {
            'len_method': self.len_method,
            'n_random_points': self.n_random_points,
            'nb_inst_per_class': self.nb_inst_per_class,
            'sel_inst_wrepl':self.sel_inst_wrepl,
            'sel_randp_wrepl':self.sel_randp_wrepl,
            'half_instance':self.half_instance,
            'half_len':self.half_len,        
            'classifier': self.classifier,
            'cand_length_list': self.cand_length_list
        }

    def init_rsast(self, X, y):
        #0- initialize variables and convert values in "y" to string
        start = time.time()
        y=np.asarray([str(x_s) for x_s in y])
        
        self.cand_length_list = {}
        self.kernel_orig_ = []
        self.kernels_generators_ = []
        self.class_generators_ = []

        list_kernels =[]
        
        
        
        n = []
        classes = np.unique(y)
        self.num_classes = classes.shape[0]
        m_kernel = 0

        #1--calculate ANOVA per each time t throught the lenght of the TS
        for i in range (X.shape[1]):
            statistic_per_class= {}
            for c in classes:
                assert len(X[np.where(y==c)[0]][:,i])> 0, 'Time t without values in TS'

                statistic_per_class[c]=X[np.where(y==c)[0]][:,i]
                #print("statistic_per_class- i:"+str(i)+', c:'+str(c))
                #print(statistic_per_class[c].shape)


            #print('Without pd series')
            #print(statistic_per_class)

            statistic_per_class=pd.Series(statistic_per_class)
            #statistic_per_class = list(statistic_per_class.values())
            # Calculate t-statistic and p-value

            try:
                t_statistic, p_value = f_oneway(*statistic_per_class)
            except DegenerateDataWarning or ConstantInputWarning:
                p_value=np.nan
            # Interpretation of the results
            # if p_value < 0.05: " The means of the populations are significantly different."
            #print('pvalue', str(p_value))
            if np.isnan(p_value):
                n.append(0)
            else:
                n.append(1-p_value)
        end = time.time()
        self.time_calculating_weights = end-start


        #2--calculate PACF and ACF for each TS chossen in each class
        start = time.time()
        for i, c in enumerate(classes):
            X_c = X[y == c]
            if self.half_instance==True:
                cnt = np.max([X_c.shape[0]//2, 1]).astype(int)
                self.nb_inst_per_class=cnt
            else:
                cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int)
            #set if the selection of instances is with replacement (if false it is not posible to select the same intance more than one)
            if self.sel_inst_wrepl ==False:
                choosen = self.random_state.permutation(X_c.shape[0])[:cnt]
            else:
                choosen = self.random_state.choice(X_c.shape[0], cnt)
            
            
            
            
            for rep, idx in enumerate(choosen):
                self.cand_length_list[c+","+str(idx)+","+str(rep)] = []
                non_zero_acf=[]
                if (self.len_method == "both" or self.len_method == "ACF" or self.len_method == "Max ACF") :
                #2.1-- Compute Autorrelation per object
                    acf_val, acf_confint = acf(X_c[idx], nlags=len(X_c[idx])-1,  alpha=.05)
                    prev_acf=0    
                    for j, conf in enumerate(acf_confint):

                        if(3<=j and (0 < acf_confint[j][0] <= acf_confint[j][1] or acf_confint[j][0] <= acf_confint[j][1] < 0) ):
                            #Consider just the maximum ACF value
                            if prev_acf!=0 and self.len_method == "Max ACF":
                                non_zero_acf.remove(prev_acf)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].remove(prev_acf)
                            non_zero_acf.append(j)
                            self.cand_length_list[c+","+str(idx)+","+str(rep)].append(j)
                            prev_acf=j        
                
                non_zero_pacf=[]
                if (self.len_method == "both" or self.len_method == "PACF" or self.len_method == "Max PACF"):
                    #2.2 Compute Partial Autorrelation per object
                    pacf_val, pacf_confint = pacf(X_c[idx], method="ols", nlags=(len(X_c[idx])//2) - 1,  alpha=.05)                
                    prev_pacf=0
                    for j, conf in enumerate(pacf_confint):

                        if(3<=j and (0 < pacf_confint[j][0] <= pacf_confint[j][1] or pacf_confint[j][0] <= pacf_confint[j][1] < 0) ):
                            #Consider just the maximum PACF value
                            if prev_pacf!=0 and self.len_method == "Max PACF":
                                non_zero_pacf.remove(prev_pacf)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].remove(prev_pacf)
                            
                            non_zero_pacf.append(j)
                            self.cand_length_list[c+","+str(idx)+","+str(rep)].append(j)
                            prev_pacf=j 
                            
                if (self.len_method == "all"):
                    self.cand_length_list[c+","+str(idx)+","+str(rep)].extend(np.arange(3,1+ len(X_c[idx])))
                
                #2.3-- Save the maximum autocorralated lag value as shapelet lenght 
                
                if len(self.cand_length_list[c+","+str(idx)+","+str(rep)])==0:
                    #chose a random lenght using the lenght of the time series (added 1 since the range start in 0)
                    rand_value= self.random_state.choice(len(X_c[idx]), 1)[0]+1
                    self.cand_length_list[c+","+str(idx)+","+str(rep)].extend([max(3,rand_value)])
                #elif len(non_zero_acf)==0:
                    #print("There is no AC in TS", idx, " of class ",c)
                #elif len(non_zero_pacf)==0:
                    #print("There is no PAC in TS", idx, " of class ",c)                 
                #else:
                    #print("There is AC and PAC in TS", idx, " of class ",c)

                #print("Kernel lenght list:",self.cand_length_list[c+","+str(idx)],"")
                 
                #remove duplicates for the list of lenghts
                self.cand_length_list[c+","+str(idx)+","+str(rep)]=list(set(self.cand_length_list[c+","+str(idx)+","+str(rep)]))
                #print("Len list:"+str(self.cand_length_list[c+","+str(idx)+","+str(rep)]))
                for max_shp_length in self.cand_length_list[c+","+str(idx)+","+str(rep)]:
                    
                    #2.4-- Choose randomly n_random_points point for a TS                
                    #2.5-- calculate the weights of probabilities for a random point in a TS
                    if sum(n) == 0 :
                        # Determine equal weights of a random point point in TS is there are no significant points
                        # print('All p values in One way ANOVA are equal to 0') 
                        weights = [1/len(n) for i in range(len(n))]
                        weights = weights[:len(X_c[idx])-max_shp_length +1]/np.sum(weights[:len(X_c[idx])-max_shp_length+1])
                    else: 
                        # Determine the weights of a random point point in TS (excluding points after n-l+1)
                        weights = n / np.sum(n)
                        weights = weights[:len(X_c[idx])-max_shp_length +1]/np.sum(weights[:len(X_c[idx])-max_shp_length+1])
                        
                    if self.half_len==True:
                        self.n_random_points=np.max([len(X_c[idx])//2, 1]).astype(int)
                    
                    
                    if self.n_random_points > len(X_c[idx])-max_shp_length+1 and self.sel_randp_wrepl==False:
                        #set a upper limit for the posible of number of random points when selecting without replacement
                        limit_rpoint=len(X_c[idx])-max_shp_length+1
                        rand_point_ts = self.random_state.choice(len(X_c[idx])-max_shp_length+1, limit_rpoint, p=weights, replace=self.sel_randp_wrepl)
                        #print("limit_rpoint:"+str(limit_rpoint))
                    else:
                        rand_point_ts = self.random_state.choice(len(X_c[idx])-max_shp_length+1, self.n_random_points, p=weights, replace=self.sel_randp_wrepl)
                        #print("n_random_points:"+str(self.n_random_points))
                    
                    #print("rpoints:"+str(rand_point_ts))
                    
                    for i in rand_point_ts:        
                        #2.6-- Extract the subsequence with that point
                        kernel = X_c[idx][i:i+max_shp_length].reshape(1,-1)
                        #print("kernel:"+str(kernel))
                        if m_kernel<max_shp_length:
                            m_kernel = max_shp_length            
                        list_kernels.append(kernel)
                        self.kernel_orig_.append(np.squeeze(kernel))
                        self.kernels_generators_.append(np.squeeze(X_c[idx].reshape(1,-1)))
                        self.class_generators_.append(c)
        
        print("total kernels:"+str(len(self.kernel_orig_)))
        
        if self.n_shapelet_samples!=None:
            print("Truncated to:"+str(self.n_shapelet_samples))
            
            self.kernel_permutated_ = self.random_state.permutation(self.kernel_orig_)[:self.n_shapelet_samples]
        else:
            self.kernel_permutated_ = self.kernel_orig_
        
        #3--save the calculated subsequences
        
        
        n_kernels = len (self.kernel_permutated_)
        
        
        self.kernels_ = np.full(
            (n_kernels, m_kernel), dtype=np.float32, fill_value=np.inf)
        
        for k, kernel in enumerate(self.kernel_permutated_):
            self.kernels_[k, :len(kernel)] = znormalize_array(kernel)
        
        end = time.time()
        self.time_creating_subsequences = end-start

    def fit(self, X, y):

        X, y = check_X_y(X, y)  # check the shape of the data

        # randomly choose reference time series and generate kernels
        self.init_rsast(X, y)

        start = time.time()
        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)
        end = time.time()
        self.transform_dataset = end-start
        
        if self.classifier is None:
            
            if X_transformed.shape[0]<=X_transformed.shape[1]: #n_features (kernels) > n_samples (intances)
                self.classifier=RidgeClassifierCV()
                print("RidgeClassifierCV:"+str("size training")+str(X_transformed.shape[0])+"<="+" kernels"+str(X_transformed.shape[1]))
            else: 
                print("LogisticRegression:"+str("size training")+str(X_transformed.shape[0])+">"+" kernels"+str(X_transformed.shape[1]))
                self.classifier=LogisticRegression()
                #self.classifier = RandomForestClassifier(min_impurity_decrease=0.05, max_features=None)

        start = time.time()
        #print('X_transformed shape')
        #print(X_transformed.shape)
        #print('X_transformed')
        #print(X_transformed)

        self.classifier.fit(X_transformed, y)  # fit the classifier
        end = time.time()
        self.time_classifier = end-start
        
        return self

    def predict(self, X):

        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels(X, self.kernels_)

        if isinstance(self.classifier, LinearClassifierMixin):
            return self.classifier._predict_proba_lr(X_transformed)
        return self.classifier.predict_proba(X_transformed)

class RSASTMF(BaseEstimator, ClassifierMixin):

    def __init__(self,n_random_points=10, nb_inst_per_class=10,max_shapelet_lengths=None, q_max=0.1, q_min=0, len_method="both", random_state=None, classifier=None, sel_inst_wrepl=False,sel_randp_wrepl=False):
        super(RSASTMF, self).__init__()
        self.n_random_points = n_random_points
        self.nb_inst_per_class = nb_inst_per_class
        self.max_shapelet_lengths = max_shapelet_lengths
        self.q_max = q_max
        self.q_min = q_min
        self.len_method = len_method
        self.random_state = np.random.RandomState(random_state) if not isinstance(
            random_state, np.random.RandomState) else random_state
        self.classifier = classifier
        self.cand_length_list = None
        self.kernels_ = None # z-normalized shapelets
        self.kernel_orig_ = None  # not z-normalized shapelets
        self.kernels_generators_ = None
        self.class_kernel_ = None
        self.dilation_kernel_ = None
        self.lambda_kernel_ = None
        self.sel_inst_wrepl=sel_inst_wrepl
        self.sel_randp_wrepl=sel_randp_wrepl
        self.time_calculating_weights = None
        self.time_creating_subsequences = None
        self.time_transform_dataset = None
        self.time_classifier = None


    def get_params(self, deep=True):
        return {
            'len_method': self.len_method,
            'n_random_points': self.n_random_points,
            'nb_inst_per_class': self.nb_inst_per_class,
            'sel_inst_wrepl':self.sel_inst_wrepl,
            'sel_randp_wrepl':self.sel_randp_wrepl,   
            'classifier': self.classifier,
            'cand_length_list': self.cand_length_list
        }

    def init_rsastmf(self, X, y):
        #0- initialize variables and convert values in "y" to string
        start = time.time()
        y=np.asarray([str(x_s) for x_s in y])
        
        self.cand_length_list = {}
        self.kernel_orig_ = []
        self.kernels_generators_ = []
        self.class_kernel_ = []
        self.dilation_kernel_ = []
        self.lambda_kernel_ = []

        
        
        
        
        n = []
        classes = np.unique(y)
        self.num_classes = classes.shape[0]
        m_kernel = 0

        #1--calculate ANOVA per each time t throught the lenght of the TS
        for i in range (X.shape[1]):
            statistic_per_class= {}
            for c in classes:
                assert len(X[np.where(y==c)[0]][:,i])> 0, 'Time t without values in TS'

                statistic_per_class[c]=X[np.where(y==c)[0]][:,i]


            statistic_per_class=pd.Series(statistic_per_class)
            #statistic_per_class = list(statistic_per_class.values())
            # Calculate t-statistic and p-value

            try:
                t_statistic, p_value = f_oneway(*statistic_per_class)
            except DegenerateDataWarning or ConstantInputWarning:
                p_value=np.nan
            # Interpretation of the results
            # if p_value < 0.05: " The means of the populations are significantly different."

            if np.isnan(p_value):
                n.append(0)
            else:
                n.append(1-p_value)
        end = time.time()
        self.time_calculating_weights = end-start


        #2--calculate PACF and ACF for each TS chossen in each class
        start = time.time()
        for i, c in enumerate(classes):
            X_c = X[y == c]
            cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int)
            #set if the selection of instances is with replacement (if false it is not posible to select the same intance more than one)
            if self.sel_inst_wrepl ==False:
                choosen = self.random_state.permutation(X_c.shape[0])[:cnt]
            else:
                choosen = self.random_state.choice(X_c.shape[0], cnt)
            
            
            
            
            for rep, idx in enumerate(choosen):
                self.cand_length_list[c+","+str(idx)+","+str(rep)] = []
                non_zero_acf=[]
                #tlt="class-"+c+",idx-"+str(idx)+",rep-"+str(rep)
                #plt.figure()
                #plt.title(tlt)
                #plt.plot(X_c[idx])
                #plt.show()
                
                if (self.len_method == "both" or self.len_method == "ACF" or self.len_method == "Max ACF") :
                #2.1-- Compute Autorrelation per object
                    acf_val, acf_confint = acf(X_c[idx], nlags=len(X_c[idx])-1,  alpha=.05)
                    
                    #plot_acf(X_c[idx],title="ACF: "+tlt, lags=len(X_c[idx])-1,  alpha=.05)                
                    #plt.show()

                    prev_acf=0    
                    for j, conf in enumerate(acf_confint):

                        if(3<=j and (0 < acf_confint[j][0] <= acf_confint[j][1] or acf_confint[j][0] <= acf_confint[j][1] < 0) ):
                            #Consider just the maximum ACF value
                            if prev_acf!=0 and self.len_method == "Max ACF":
                                non_zero_acf.remove(prev_acf)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].remove(prev_acf)
                            non_zero_acf.append(j)
                            self.cand_length_list[c+","+str(idx)+","+str(rep)].append(j)
                            prev_acf=j        
                
                non_zero_pacf=[]
                if (self.len_method == "both" or self.len_method == "PACF" or self.len_method == "Max PACF"):
                    #2.2 Compute Partial Autorrelation per object
                    pacf_val, pacf_confint = pacf(X_c[idx], method="ols", nlags=(len(X_c[idx])//2) - 1,  alpha=.05)

                    
                    #plot_pacf(X_c[idx],title="PACF: "+tlt, method="ols", lags=(len(X_c[idx])//2) - 1,  alpha=.05)                
                    #plt.show()
                    prev_pacf=0
                    for j, conf in enumerate(pacf_confint):

                        if(3<=j and (0 < pacf_confint[j][0] <= pacf_confint[j][1] or pacf_confint[j][0] <= pacf_confint[j][1] < 0) ):
                            #Consider just the maximum PACF value
                            if prev_pacf!=0 and self.len_method == "Max PACF":
                                non_zero_pacf.remove(prev_pacf)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].remove(prev_pacf)
                            #print("Truncated lengths to:"+str(self.max_shapelet_lengths))
                            if self.max_shapelet_lengths!=None and (self.max_shapelet_lengths > len(non_zero_pacf)):
                                
                                non_zero_pacf.append(j)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].append(j)
                                prev_pacf=j 
                            
                if (self.len_method == "all"):
                    self.cand_length_list[c+","+str(idx)+","+str(rep)].extend(np.arange(3,1+ len(X_c[idx])))
                
                #2.3-- Save the maximum autocorralated lag value as shapelet lenght 
                
                if len(self.cand_length_list[c+","+str(idx)+","+str(rep)])==0:
                    #chose a random lenght using the lenght of the time series (added 1 since the range start in 0)
                    rand_value= self.random_state.choice(len(X_c[idx]), 1)[0]+1
                    self.cand_length_list[c+","+str(idx)+","+str(rep)].extend([max(3,rand_value)])

                #remove duplicates for the list of lenghts
                self.cand_length_list[c+","+str(idx)+","+str(rep)]=list(set(self.cand_length_list[c+","+str(idx)+","+str(rep)]))
                for max_shp_length in self.cand_length_list[c+","+str(idx)+","+str(rep)]:
                    
                    #2.4-- Choose randomly n_random_points point for a TS                
                    #2.5-- calculate the weights of probabilities for a random point in a TS
                    if sum(n) == 0 :
                        # Determine equal weights of a random point point in TS is there are no significant points
                        # print('All p values in One way ANOVA are equal to 0') 
                        weights = [1/len(n) for i in range(len(n))]
                        weights = weights[:len(X_c[idx])-max_shp_length +1]/np.sum(weights[:len(X_c[idx])-max_shp_length+1])
                    else: 
                        # Determine the weights of a random point point in TS (excluding points after n-l+1)
                        weights = n / np.sum(n)
                        weights = weights[:len(X_c[idx])-max_shp_length +1]/np.sum(weights[:len(X_c[idx])-max_shp_length+1])
                                           
                    
                    if self.n_random_points > len(X_c[idx])-max_shp_length+1 and self.sel_randp_wrepl==False:
                        #set a upper limit for the posible of number of random points when selecting without replacement
                        limit_rpoint=len(X_c[idx])-max_shp_length+1
                        rand_point_ts = self.random_state.choice(len(X_c[idx])-max_shp_length+1, limit_rpoint, p=weights, replace=self.sel_randp_wrepl)

                    else:
                        rand_point_ts = self.random_state.choice(len(X_c[idx])-max_shp_length+1, self.n_random_points, p=weights, replace=self.sel_randp_wrepl)

                    
                    
                    
                    for i in rand_point_ts:        
                        #max_shp_length=5      
                        x=random.uniform(0, math.log2(len(X_c[idx]) / max_shp_length))
                        shp_dil=int(2**x)
                        #shp_dil=1
                        max_shp_length=max_shp_length*shp_dil
                        #print("dil: 2**"+str(x)+"="+str(int(2**x)))
                        #print("orig kernel-len "+str((max_shp_length))+":"+str(X_c[idx][i:i+max_shp_length]))
                        
                        #2.6-- Extract the subsequence with that point
                        kernel = X_c[idx][i:i+max_shp_length].reshape(1,-1)
                        kernel = np.squeeze(kernel)
                        #if shp_dil>1: 
                            #print("orig kernel-len "+str(len(kernel))+":"+str(kernel))
                        
                        for j in range(len(kernel)):
                            
                            if(j%shp_dil==0):
                                kernel[j]=kernel[j]
                            else: 
                                kernel[j]=np.nan

                        #if shp_dil>1:    
                            #print("dil kernel-len "+str(max_shp_length)+" shp_dil "+str(shp_dil)+":"+str(kernel))
                        if m_kernel<max_shp_length:
                            m_kernel = max_shp_length            
                        
                        #choosen = self.random_state.choice(X_c.shape[0], 1)[0]
                        ld = get_lambda_rdst(X_c[idx], kernel, self.q_max, self.q_min)

                        self.kernel_orig_.append(kernel)
                        self.kernels_generators_.append(np.squeeze(X_c[idx].reshape(1,-1)))
                        self.class_kernel_.append(c)
                        self.dilation_kernel_.append(shp_dil)
                        self.lambda_kernel_.append(ld)
        
        print("total kernels:"+str(len(self.kernel_orig_)))
        
        self.kernel_orig_ = self.kernel_orig_
        
        #3--save the calculated subsequences
        
        
        n_kernels = len (self.kernel_orig_)
        
        
        self.kernels_ = np.full(
            (n_kernels, m_kernel), dtype=np.float32, fill_value=np.inf)
        
        for k, kernel in enumerate(self.kernel_orig_):
            self.kernels_[k, :len(kernel)] = znormalize_array(kernel)
        
        end = time.time()
        self.time_creating_subsequences = end-start

    def fit(self, X, y):

        X, y = check_X_y(X, y)  # check the shape of the data

        # randomly choose reference time series and generate kernels
        self.init_rsastmf(X, y)

        start = time.time()
        # subsequence transform of X
        X_transformed = apply_kernels_mf(X, self.kernels_, self.lambda_kernel_)
        end = time.time()
        self.transform_dataset = end-start

        if self.classifier is None:
            
            if X_transformed.shape[0]<=X_transformed.shape[1]: #n_features (kernels) > n_samples (intances)
                self.classifier=RidgeClassifierCV()
                print("RidgeClassifierCV:"+str("size training")+str(X_transformed.shape[0])+"<="+" kernels"+str(X_transformed.shape[1]))
            else: 
                print("LogisticRegression:"+str("size training")+str(X_transformed.shape[0])+">"+" kernels"+str(X_transformed.shape[1]))
                self.classifier=LogisticRegression()
                #self.classifier = RandomForestClassifier(min_impurity_decrease=0.05, max_features=None)

        start = time.time()
        #print('X_transformed shape')
        #print(X_transformed.shape)
        #print("Transformed ds in training")
        #print(pd.DataFrame(X_transformed))


        self.classifier.fit(X_transformed, y)  # fit the classifier
        end = time.time()
        self.time_classifier = end-start
        
        return self

    def predict(self, X):

        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels_mf(X, self.kernels_, self.lambda_kernel_)

        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels_mf(X, self.kernels_, self.lambda_kernel_)

        if isinstance(self.classifier, LinearClassifierMixin):
            return self.classifier._predict_proba_lr(X_transformed)
        return self.classifier.predict_proba(X_transformed)
    
    def plot_most_important_features_mfrsast(self, limit = 3, scale_color=False):
        type_features_cl=["min","argmin","SO"]*len(self.kernel_orig_)
        features = zip(self.kernel_orig_, self.classifier.coef_[0], self.dilation_kernel_, type_features_cl)
        #sort features by absolute score in classifier
        sorted_features = sorted(features, key=lambda sublist: abs(sublist[1]), reverse=True)
        
        for l, sf in enumerate(sorted_features[:limit]):
            
            kernel, score, dilation,t_feature = sf
            print("kernel-" +str(l+1)+":"+str(kernel))

            dmask = np.isfinite(kernel.astype(np.double))
            shp_range=np.arange(kernel.size)
            if scale_color:
                
                plt.plot(shp_range[dmask], kernel[dmask], linewidth=50*score, label="feature"+str(l+1)+": "+"d="+str(dilation)+" coef="+str(f'{score:.5}'), linestyle='-', marker='o')
            else:
                
                plt.plot(shp_range[dmask], kernel[dmask], label="feature"+str(l+1)+": "+"d="+str(dilation)+" coef="+str(f'{score:.5}'), linestyle='-', marker='o')
        plt.legend()
        plt.show()

    def plot_most_important_feature_on_ts_mfrsast( self, offset=0, limit = 3, fname=None, znormalized=False):
        '''Plot the most important features on ts'''
                
        type_features_cl=["min","argmin","SO"]*len(self.kernel_orig_)
        features = zip(self.kernel_orig_, self.classifier.coef_[0], self.dilation_kernel_, type_features_cl, self.kernels_generators_, self.class_kernel_)
        
        sorted_features = sorted(features, key=lambda sublist: abs(sublist[1]), reverse=True)
        max_ = min(limit, len(sorted_features) - offset)    
        #sorted_features = sorted(features, key=itemgetter(1), reverse=True)
        
        
        
        if max_ <= 0:
            print('Nothing to plot')
            return        
        
        
        for s, l in enumerate(np.unique(self.class_kernel_)):
            fig, axes = plt.subplots(1, max_, sharey=True, figsize=(3*max_, 3), tight_layout=True, clear=True)
            
                    
            for f in range(max_):
                
                kernel, score, dilation, type_f, ts, label = sorted_features[f+offset]
                
                if label!=l:
                    *_, ts, _=list(filter(lambda x: x[5] == l,sorted_features))[0]

                    
                    

                kernel_d=[]
                for value in kernel:
                    for j in range(dilation):
                        if j==0:
                            kernel_d.append(value)        
                        else:
                            kernel_d.append(None)
                kernel_d=np.array(kernel_d)
                
                if znormalized:
                    kernel_d = znormalize_array(kernel_d)
                    ts = znormalize_array(ts)            
                
                d_best = np.inf

                
                for i in range(ts.size - kernel_d.size + 1):

                    d=0
                    for k, value in enumerate(kernel_d):
                    

                        
                        if kernel_d[k] is not None:
                            d = d+(ts[i:i+kernel_d.size][k] - kernel_d[k])**2
                        else:
                            break
                    if d < d_best:
                        d_best = d
                        start_pos = i
                dmask = np.isfinite(kernel_d.astype(np.double))
                shp_range=np.arange(start_pos, start_pos + kernel_d.size)
                axes[f].plot(shp_range[dmask], kernel_d[dmask], linewidth=6,color="darkorange", linestyle='-', marker='o')
                axes[f].plot(range(ts.size), ts, linewidth=2,color='darkblue')
                axes[f].set_title(f'feature: {f+1+offset}, type: {type_f}')
                #print('gph shapelet values:',str(f+1),' start_pos:',start_pos,' shape:', kernel_d.size,' dilation:', str(dilation))
                #print(" shapelet:", kernel_d )

            fig.suptitle(f'Ground truth class: {l}', fontsize=15)

            plt.show()

            if fname is not None:
                fig.savefig(fname)

class DICTRSAST(BaseEstimator, ClassifierMixin):

    def __init__(self,n_random_points=10, nb_inst_per_class=10,max_shapelet_lengths=None, q_max=0.1, q_min=0, len_method="both", random_state=None, classifier=None, sel_inst_wrepl=False,sel_randp_wrepl=False):
        super(DICTRSAST, self).__init__()
        self.n_random_points = n_random_points
        self.nb_inst_per_class = nb_inst_per_class
        self.max_shapelet_lengths = max_shapelet_lengths
        self.q_max = q_max
        self.q_min = q_min
        self.len_method = len_method
        self.random_state = np.random.RandomState(random_state) if not isinstance(
            random_state, np.random.RandomState) else random_state
        self.classifier = classifier
        self.cand_length_list = None
        self.kernels_ = None # z-normalized shapelets
        self.kernel_orig_ = None  # not z-normalized shapelets
        self.kernels_generators_ = None
        self.class_kernel_ = None
        self.dilation_kernel_ = None
        self.sel_inst_wrepl=sel_inst_wrepl
        self.sel_randp_wrepl=sel_randp_wrepl
        self.time_calculating_weights = None
        self.time_creating_subsequences = None
        self.time_transform_dataset = None
        self.time_classifier = None


    def get_params(self, deep=True):
        return {
            'len_method': self.len_method,
            'n_random_points': self.n_random_points,
            'nb_inst_per_class': self.nb_inst_per_class,
            'sel_inst_wrepl':self.sel_inst_wrepl,
            'sel_randp_wrepl':self.sel_randp_wrepl,   
            'classifier': self.classifier,
            'cand_length_list': self.cand_length_list
        }

    def init_dictrsast(self, X, y):
        #0- initialize variables and convert values in "y" to string
        start = time.time()
        y=np.asarray([str(x_s) for x_s in y])
        
        self.cand_length_list = {}
        self.kernel_orig_ = []
        self.kernels_generators_ = []
        self.class_kernel_ = []
        self.dilation_kernel_ = []

        
        
        
        
        n = []
        classes = np.unique(y)
        self.num_classes = classes.shape[0]
        m_kernel = 0

        #1--calculate ANOVA per each time t throught the lenght of the TS
        for i in range (X.shape[1]):
            statistic_per_class= {}
            for c in classes:
                assert len(X[np.where(y==c)[0]][:,i])> 0, 'Time t without values in TS'

                statistic_per_class[c]=X[np.where(y==c)[0]][:,i]


            statistic_per_class=pd.Series(statistic_per_class)
            #statistic_per_class = list(statistic_per_class.values())
            # Calculate t-statistic and p-value

            try:
                t_statistic, p_value = f_oneway(*statistic_per_class)
            except DegenerateDataWarning or ConstantInputWarning:
                p_value=np.nan
            # Interpretation of the results
            # if p_value < 0.05: " The means of the populations are significantly different."

            if np.isnan(p_value):
                n.append(0)
            else:
                n.append(1-p_value)
        end = time.time()
        self.time_calculating_weights = end-start


        #2--calculate PACF and ACF for each TS chossen in each class
        start = time.time()
        for i, c in enumerate(classes):
            X_c = X[y == c]
            cnt = np.min([self.nb_inst_per_class, X_c.shape[0]]).astype(int)
            #set if the selection of instances is with replacement (if false it is not posible to select the same intance more than one)
            if self.sel_inst_wrepl ==False:
                choosen = self.random_state.permutation(X_c.shape[0])[:cnt]
            else:
                choosen = self.random_state.choice(X_c.shape[0], cnt)
            
            
            
            
            for rep, idx in enumerate(choosen):
                self.cand_length_list[c+","+str(idx)+","+str(rep)] = []
                non_zero_acf=[]
                tlt="class-"+c+",idx-"+str(idx)+",rep-"+str(rep)
                #plt.figure()
                #plt.title(tlt)
                #plt.plot(X_c[idx])
                #plt.show()
                
                if (self.len_method == "both" or self.len_method == "ACF" or self.len_method == "Max ACF") :
                #2.1-- Compute Autorrelation per object
                    acf_val, acf_confint = acf(X_c[idx], nlags=len(X_c[idx])-1,  alpha=.05)
                    
                    #plot_acf(X_c[idx],title="ACF: "+tlt, lags=len(X_c[idx])-1,  alpha=.05)                
                    plt.show()

                    prev_acf=0    
                    for j, conf in enumerate(acf_confint):

                        if(3<=j and (0 < acf_confint[j][0] <= acf_confint[j][1] or acf_confint[j][0] <= acf_confint[j][1] < 0) ):
                            #Consider just the maximum ACF value
                            if prev_acf!=0 and self.len_method == "Max ACF":
                                non_zero_acf.remove(prev_acf)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].remove(prev_acf)
                            non_zero_acf.append(j)
                            self.cand_length_list[c+","+str(idx)+","+str(rep)].append(j)
                            prev_acf=j        
                
                non_zero_pacf=[]
                if (self.len_method == "both" or self.len_method == "PACF" or self.len_method == "Max PACF"):
                    #2.2 Compute Partial Autorrelation per object
                    pacf_val, pacf_confint = pacf(X_c[idx], method="ols", nlags=(len(X_c[idx])//2) - 1,  alpha=.05)

                    
                    #plot_pacf(X_c[idx],title="PACF: "+tlt, method="ols", lags=(len(X_c[idx])//2) - 1,  alpha=.05)                
                    plt.show()
                    prev_pacf=0
                    for j, conf in enumerate(pacf_confint):

                        if(3<=j and (0 < pacf_confint[j][0] <= pacf_confint[j][1] or pacf_confint[j][0] <= pacf_confint[j][1] < 0) ):
                            #Consider just the maximum PACF value
                            if prev_pacf!=0 and self.len_method == "Max PACF":
                                non_zero_pacf.remove(prev_pacf)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].remove(prev_pacf)
                            #print("Truncated lengths to:"+str(self.max_shapelet_lengths))
                            if self.max_shapelet_lengths!=None and (self.max_shapelet_lengths > len(non_zero_pacf)):
                                
                                non_zero_pacf.append(j)
                                self.cand_length_list[c+","+str(idx)+","+str(rep)].append(j)
                                prev_pacf=j 
                            
                if (self.len_method == "all"):
                    self.cand_length_list[c+","+str(idx)+","+str(rep)].extend(np.arange(3,1+ len(X_c[idx])))
                
                #2.3-- Save the maximum autocorralated lag value as shapelet lenght 
                
                if len(self.cand_length_list[c+","+str(idx)+","+str(rep)])==0:
                    #chose a random lenght using the lenght of the time series (added 1 since the range start in 0)
                    rand_value= self.random_state.choice(len(X_c[idx]), 1)[0]+1
                    self.cand_length_list[c+","+str(idx)+","+str(rep)].extend([max(3,rand_value)])

                #remove duplicates for the list of lenghts
                self.cand_length_list[c+","+str(idx)+","+str(rep)]=list(set(self.cand_length_list[c+","+str(idx)+","+str(rep)]))
                for max_shp_length in self.cand_length_list[c+","+str(idx)+","+str(rep)]:
                    
                    #2.4-- Choose randomly n_random_points point for a TS                
                    #2.5-- calculate the weights of probabilities for a random point in a TS
                    if sum(n) == 0 :
                        # Determine equal weights of a random point point in TS is there are no significant points
                        # print('All p values in One way ANOVA are equal to 0') 
                        weights = [1/len(n) for i in range(len(n))]
                        weights = weights[:len(X_c[idx])-max_shp_length +1]/np.sum(weights[:len(X_c[idx])-max_shp_length+1])
                    else: 
                        # Determine the weights of a random point point in TS (excluding points after n-l+1)
                        weights = n / np.sum(n)
                        weights = weights[:len(X_c[idx])-max_shp_length +1]/np.sum(weights[:len(X_c[idx])-max_shp_length+1])
                                           
                    
                    if self.n_random_points > len(X_c[idx])-max_shp_length+1 and self.sel_randp_wrepl==False:
                        #set a upper limit for the posible of number of random points when selecting without replacement
                        limit_rpoint=len(X_c[idx])-max_shp_length+1
                        rand_point_ts = self.random_state.choice(len(X_c[idx])-max_shp_length+1, limit_rpoint, p=weights, replace=self.sel_randp_wrepl)

                    else:
                        rand_point_ts = self.random_state.choice(len(X_c[idx])-max_shp_length+1, self.n_random_points, p=weights, replace=self.sel_randp_wrepl)

                    
                    
                    
                    for i in rand_point_ts:  
                        #max_shp_length=5      
                        x=random.uniform(0, math.log2(len(X_c[idx]) / max_shp_length))
                        shp_dil=int(2**x)
                        #shp_dil=1
                        max_shp_length=max_shp_length*shp_dil
                        #print("dil: 2**"+str(x)+"="+str(int(2**x)))
                        #print("orig kernel-len "+str((max_shp_length))+":"+str(X_c[idx][i:i+max_shp_length]))
                        #2.6-- Extract the subsequence with that point
                        
                        kernel = X_c[idx][i:i+max_shp_length].reshape(1,-1)
                        kernel = np.squeeze(kernel)
                        
                        for j in range(len(kernel)):
                            
                            if(j%shp_dil==0):
                                kernel[j]=kernel[j]
                            else: 
                                kernel[j]=np.nan

                        #if shp_dil>1:    
                            #print("dil kernel-len "+str(max_shp_length)+" shp_dil "+str(shp_dil)+":"+str(kernel))
                        if m_kernel<max_shp_length:
                            m_kernel = max_shp_length        
                        
                        self.kernel_orig_.append(kernel)
                        self.kernels_generators_.append(np.squeeze(X_c[idx].reshape(1,-1)))
                        self.class_kernel_.append(c)
                        self.dilation_kernel_.append(shp_dil)
        
        print("total kernels:"+str(len(self.kernel_orig_)))
        
        self.kernel_orig_ = self.kernel_orig_
        
        #3--save the calculated subsequences
        
        
        n_kernels = len (self.kernel_orig_)
        
        
        self.kernels_ = np.full(
            (n_kernels, m_kernel), dtype=np.float32, fill_value=np.inf)
        
        for k, kernel in enumerate(self.kernel_orig_):
            self.kernels_[k, :len(kernel)] = znormalize_array(kernel)
        
        end = time.time()
        self.time_creating_subsequences = end-start

    def fit(self, X, y):

        X, y = check_X_y(X, y)  # check the shape of the data

        # randomly choose reference time series and generate kernels
        self.init_dictrsast(X, y)

        start = time.time()
        # subsequence transform of X
        X_transformed = apply_kernels_dict(X, self.kernels_, self.q_max, self.q_min)
        end = time.time()
        self.transform_dataset = end-start

        if self.classifier is None:
            
            if X_transformed.shape[0]<=X_transformed.shape[1]: #n_features (kernels) > n_samples (intances)
                self.classifier=RidgeClassifierCV()
                print("RidgeClassifierCV:"+str("size training")+str(X_transformed.shape[0])+"<="+" kernels"+str(X_transformed.shape[1]))
            else: 
                print("LogisticRegression:"+str("size training")+str(X_transformed.shape[0])+">"+" kernels"+str(X_transformed.shape[1]))
                self.classifier=LogisticRegression()
                #self.classifier = RandomForestClassifier(min_impurity_decrease=0.05, max_features=None)

        start = time.time()
        #print('X_transformed shape')
        #print(X_transformed.shape)
        #print("Transformed ds in training")
        #print(pd.DataFrame(X_transformed))


        self.classifier.fit(X_transformed, y)  # fit the classifier
        end = time.time()
        self.time_classifier = end-start
        
        return self

    def predict(self, X):

        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels_dict(X, self.kernels_, self.q_max, self.q_min)

        return self.classifier.predict(X_transformed)

    def predict_proba(self, X):
        check_is_fitted(self)  # make sure the classifier is fitted

        X = check_array(X)  # validate the shape of X

        # subsequence transform of X
        X_transformed = apply_kernels_dict(X, self.kernels_, self.q_max, self.q_min)

        if isinstance(self.classifier, LinearClassifierMixin):
            return self.classifier._predict_proba_lr(X_transformed)
        return self.classifier.predict_proba(X_transformed)
    
    def plot_most_important_features_mfrsast(self, limit = 3, scale_color=False):
        type_features_cl=["SO"]*len(self.kernel_orig_)
        features = zip(self.kernel_orig_, self.classifier.coef_[0], self.dilation_kernel_, type_features_cl)
        #sort features by absolute score in classifier
        sorted_features = sorted(features, key=lambda sublist: abs(sublist[1]), reverse=True)
        
        for l, sf in enumerate(sorted_features[:limit]):
            
            kernel, score, dilation,t_feature = sf
            print("kernel-" +str(l+1)+":"+str(kernel))

            dmask = np.isfinite(kernel.astype(np.double))
            shp_range=np.arange(kernel.size)
            if scale_color:
                
                plt.plot(shp_range[dmask], kernel[dmask], linewidth=50*score, label="feature"+str(l+1)+": "+"d="+str(dilation)+" coef="+str(f'{score:.5}'), linestyle='-', marker='o')
            else:
                
                plt.plot(shp_range[dmask], kernel[dmask], label="feature"+str(l+1)+": "+"d="+str(dilation)+" coef="+str(f'{score:.5}'), linestyle='-', marker='o')
        plt.legend()
        plt.show()

    def plot_most_important_feature_on_ts_mfrsast( self, offset=0, limit = 3, fname=None, znormalized=False):
        '''Plot the most important features on ts'''
                
        type_features_cl=["SO"]*len(self.kernel_orig_)
        features = zip(self.kernel_orig_, self.classifier.coef_[0], self.dilation_kernel_, type_features_cl, self.kernels_generators_, self.class_kernel_)
        
        sorted_features = sorted(features, key=lambda sublist: abs(sublist[1]), reverse=True)
        max_ = min(limit, len(sorted_features) - offset)    
        #sorted_features = sorted(features, key=itemgetter(1), reverse=True)
        
        
        
        if max_ <= 0:
            print('Nothing to plot')
            return        
        
        
        for s, l in enumerate(np.unique(self.class_kernel_)):
            fig, axes = plt.subplots(1, max_, sharey=True, figsize=(3*max_, 3), tight_layout=True, clear=True)
            
                    
            for f in range(max_):
                
                kernel, score, dilation, type_f, ts, label = sorted_features[f+offset]
                
                if label!=l:
                    *_, ts, _=list(filter(lambda x: x[5] == l,sorted_features))[0]

                    
                    

                kernel_d=[]
                for value in kernel:
                    for j in range(dilation):
                        if j==0:
                            kernel_d.append(value)        
                        else:
                            kernel_d.append(None)
                kernel_d=np.array(kernel_d)
                
                if znormalized:
                    kernel_d = znormalize_array(kernel_d)
                    ts = znormalize_array(ts)            
                
                d_best = np.inf

                
                for i in range(ts.size - kernel_d.size + 1):

                    d=0
                    for k, value in enumerate(kernel_d):
                    

                        
                        if kernel_d[k] is not None:
                            d = d+(ts[i:i+kernel_d.size][k] - kernel_d[k])**2
                        else:
                            break
                    if d < d_best:
                        d_best = d
                        start_pos = i
                dmask = np.isfinite(kernel_d.astype(np.double))
                shp_range=np.arange(start_pos, start_pos + kernel_d.size)
                axes[f].plot(shp_range[dmask], kernel_d[dmask], linewidth=6,color="darkorange", linestyle='-', marker='o')
                axes[f].plot(range(ts.size), ts, linewidth=2,color='darkblue')
                axes[f].set_title(f'feature: {f+1+offset}, type: {type_f}')
                #print('gph shapelet values:',str(f+1),' start_pos:',start_pos,' shape:', kernel_d.size,' dilation:', str(dilation))
                #print(" shapelet:", kernel_d )

            fig.suptitle(f'Ground truth class: {l}', fontsize=15)

            plt.show();

            if fname is not None:
                fig.savefig(fname)

if __name__ == "__main__":

    ds='Fungi' # Chosing a dataset from # Number of classes to consider

    rtype="numpy2D"
    
    #X_train, y_train = load_UCR_UEA_dataset(name=ds, split="train",extract_path="data", return_type=rtype)
    
    
    #X_train=np.nan_to_num(X_train)
    #y_train=np.nan_to_num(y_train)
    
    #X_test, y_test = load_UCR_UEA_dataset(name=ds, split="test", extract_path="data", return_type=rtype)
    
    #X_test=np.nan_to_num(X_test)
    #y_test=np.nan_to_num(y_test)
    #print('Format: load_UCR_UEA_dataset')
    #print(X_train.shape)
    #print(X_test.shape)
    #print(y_train.shape)
    #print(y_test.shape)


    #y_train = list(map(int, y_train))
    #y_test =list(map(int, y_test))    
    #print(X_train[0])  
     
    """
    print("ds:"+ds)
    X_train_mod=[]
    for i , element in enumerate(X_train):
        element=np.array(element[0])
        print("TS N:"+str(i)+" len:"+str(element.shape))
        #print(element)
        X_train_mod.append(element)
       
    X_train_mod= np.array(X_train_mod)
    print(X_train_mod.shape) 
    
    X_train_mod=np.nan_to_num(X_train_mod)
    """

    path="/home/nicolas/rsast_mf/rsast_mf/data/"
    ds_train_lds , ds_test_lds = load_dataset(ds_folder=path,ds_name=ds,shuffle=False)
    X_test_lds, y_test_lds = format_dataset(ds_test_lds)
    X_train_lds, y_train_lds = format_dataset(ds_train_lds)
    
    X_train_lds=np.nan_to_num(X_train_lds)
    y_train_lds=np.nan_to_num(y_train_lds)
    X_test_lds=np.nan_to_num(X_test_lds)
    y_test_lds=np.nan_to_num(y_test_lds)
    
    print('Format: load_dataset_'+ds)
    print(X_train_lds.shape)
    print(X_train_lds[0].shape)
    print(X_train_lds[1].shape)
    print(X_test_lds.shape)
    

    print(y_train_lds.shape)
    print(y_test_lds.shape)
    

   
    
    start = time.time()
    random_state = None
    rsast_ridge = RSAST(n_random_points=10, nb_inst_per_class=10, len_method="both")
    rsast_ridge.fit(X_train_lds, y_train_lds)
    end = time.time()
    print('rsast score :', rsast_ridge.score(X_test_lds, y_test_lds))
    print('duration:', end-start)
    print('params:', rsast_ridge.get_params()) 
    #plot_most_important_feature_on_ts(set_ts=rsast_ridge.kernels_generators_, labels=rsast_ridge.class_generators_, features=rsast_ridge.kernel_orig_, scores=rsast_ridge.classifier.coef_[0], limit=3, offset=0,znormalized=False)   
    #plot_most_important_features(rsast_ridge.kernel_orig_, rsast_ridge.classifier.coef_[0], limit=3,scale_color=False)

    
    start = time.time()
    random_state = None
    rsastmf_ridge = RSASTMF(n_random_points=10, nb_inst_per_class=10, len_method="both", q_max=0.1, q_min=0)
    rsastmf_ridge.fit(X_train_lds, y_train_lds)
    end = time.time()
    print('rsastmf score :', rsastmf_ridge.score(X_test_lds, y_test_lds))
    print('duration:', end-start)
    print('params:', rsastmf_ridge.get_params()) 
    #print('classifier:',rsast_ridge.classifier.coef_[0])
    
    #fname = f'images/chinatown-rf-class{c}-top5-features-on-ref-ts.jpg'
    #print(f"ts.shape{pd.array(rsast_ridge.kernels_generators_).shape}")
    #print(f"kernel_d.shape{pd.array(rsast_ridge.kernel_orig_).shape}")
    #rsastmf_ridge.plot_most_important_features_mfrsast()
    #rsastmf_ridge.plot_most_important_feature_on_ts_mfrsast()

    start = time.time()
    random_state = None
    dictrsast_ridge = DICTRSAST(n_random_points=10, nb_inst_per_class=10, len_method="both", q_max=0.1, q_min=0)
    dictrsast_ridge.fit(X_train_lds, y_train_lds)
    end = time.time()
    print('dictrsast score :', dictrsast_ridge.score(X_test_lds, y_test_lds))
    print('duration:', end-start)
    print('params:', dictrsast_ridge.get_params()) 
    """
    """
    X_train = X_train_lds[:, np.newaxis, :]
    X_test = X_test_lds[:, np.newaxis, :]
    y_train=np.asarray([int(x_s) for x_s in y_train_lds])
    y_test=np.asarray([int(x_s) for x_s in y_test_lds])
    start = time.time()

    rdst = RDSTClassifier(
        max_shapelets=4,
        shapelet_lengths=[7],
        proba_normalization=0,
        save_transformed_data=True
    )
    rdst = RDSTClassifier(proba_normalization=0, save_transformed_data=True)
    rdst.fit(X_train, y_train)
    end = time.time()
    

    
    print('rdst score :', rdst.score(X_test, y_test))
    print('duration:', end-start)
    print('params:', rdst.get_params())
    #print(rdst.transformed_data_)
    """
    for i, shp in enumerate(rdst._transformer.shapelets_[0].squeeze()):
        print('rdst shapelet values:',str(i+1)," shape:", shp.shape," shapelet:", shp )
    
    for i, dilation in enumerate(rdst._transformer.shapelets_[2].squeeze()):
        print('rdst dilation parameter:',str(i+1)," shape:", shp.shape," dilation:", dilation )
    
    for i, treshold in enumerate(rdst._transformer.shapelets_[3].squeeze()):
        print('rdst treshold parameter:',str(i+1)," shape:", shp.shape," treshold:", treshold )
   
    for i, normalization in enumerate(rdst._transformer.shapelets_[4].squeeze()):
        print('rdst normalization parameter:',str(i+1)," shape:", shp.shape," normalization:", normalization )
    
    for i, coef in enumerate(rdst._estimator["ridgeclassifiercv"].coef_):
        print('rdst coef:',str(i+1)," shape:", coef.shape," coef:", coef )
    """
    """
    features_cl=rdst._transformer.shapelets_[0].squeeze()
    dilations_cl=rdst._transformer.shapelets_[2].squeeze()
    
    coef_cl=rdst._estimator["ridgeclassifiercv"].coef_[0]
    features_cl=[a for a in features_cl for i in range(3)]
    dilations_cl=[a for a in dilations_cl for i in range(3)]
    type_features_cl=["min","argmin","SO"]*len(features_cl)

    for l in pd.unique(rsast_ridge.class_generators_):
        
        all=zip(rsast_ridge.kernels_generators_,rsast_ridge.class_generators_)
        
        ts_cl=list(filter(lambda x: x[1]==l,all))[0][0]
        ts_cl=[ts_cl for i in range(len(features_cl))]
        labels=[l for i in range(len(features_cl))]
        plot_most_important_feature_on_ts(set_ts=ts_cl, labels=labels, features=features_cl, scores=coef_cl,dilations=dilations_cl,type_features=type_features_cl, limit=3, offset=0,znormalized=False)   
    plot_most_important_features(features_cl, coef_cl, dilations=dilations_cl, limit=3, scale_color=False)
    
    min_shp_length = 3
    max_shp_length = X_train_lds.shape[1]
    candidate_lengths = np.arange(min_shp_length, max_shp_length+1)
    # candidate_lengths = (3, 7, 9, 11)
    nb_inst_per_class = 1
    ridge = RidgeClassifierCV(alphas = np.logspace(-3, 3, 10))
    
    start = time.time()
    random_state = None 
    sast_ridge = SAST(cand_length_list=candidate_lengths,
                          nb_inst_per_class=nb_inst_per_class, 
                          random_state=random_state, classifier=ridge)
    sast_ridge.fit(X_train_lds, y_train_lds)
    end = time.time()
    print('sast score :', sast_ridge.score(X_test_lds, y_test_lds))
    print('duration:', end-start)
    print('params:', sast_ridge.get_params()) 
    #print('classifier:',rsast_ridge.classifier.coef_[0])
    
    #fname = f'images/chinatown-rf-class{c}-top5-features-on-ref-ts.jpg'
    #print(f"ts.shape{pd.array(rsast_ridge.kernels_generators_).shape}")
    #print(f"kernel_d.shape{pd.array(rsast_ridge.kernel_orig_).shape}")
    for c, ts in sast_ridge.kernels_generators_.items():
        plot_most_important_feature_sast_on_ts(ts.squeeze(), c, sast_ridge.kernel_orig_, sast_ridge.classifier.coef_[0], limit=3, offset=0) # plot only the first model one-vs-all model's features
    """
