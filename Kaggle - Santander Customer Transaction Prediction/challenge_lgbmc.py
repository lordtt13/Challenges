# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 14:16:54 2019

@author: tanma
"""

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import KFold
from lightgbm import LGBMClassifier
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier,AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import accuracy_score,roc_auc_score
import os
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA

data = pd.read_csv("C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challlenge (Kaggle)/train.csv")
test = pd.read_csv("C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challlenge (Kaggle)/test.csv")

x = data.drop(['ID_code','target'], axis = 1)
y = data['target']
features = x.columns



param = {
        'bagging_freq': 5,
        'bagging_fraction': 0.32,
        'boost_from_average':'false',
        'feature_fraction': 0.045,
        'learning_rate': 0.01,
        'max_depth': -1,  
        'metric':'auc',
        'min_data_in_leaf': 80,
        'min_sum_hessian_in_leaf': 10.0,
        'num_leaves': 13,
        'num_threads': 8,
        'tree_learner': 'serial',
        'lambda_l2':0.1,
        'objective': 'binary', 
        'verbosity': 1,
        'boosting_type': 'gbdt'
    }

num_round = 1000000
l = list(range(1,35))
pip_lasso = Pipeline([('pca',PCA()),('lgbm',XGBClassifier())])
param_grid = {'pca__n_components':l}
gbr = GridSearchCV(pip_lasso,param_grid).fit(x,y)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(gbr, x, y, cv=5)
