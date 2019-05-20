# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:58:34 2019

@author: tanma
"""

import numpy as np 
import pandas as pd 
from sklearn.preprocessing import MinMaxScaler, Imputer
from sklearn.model_selection import KFold
import lightgbm as lgb
from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score
import os

data = pd.read_csv("C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challenge (Kaggle)/train.csv")
test = pd.read_csv("C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challenge (Kaggle)/test.csv")

x = data.drop(['ID_code','target'], axis = 1)
y = data['target']
features = x.columns

scaler = MinMaxScaler()
x = scaler.fit_transform(x)

x = pd.DataFrame(x)
x = x[(x>0.05)&(x<0.95)]

imputer = Imputer()
x = imputer.fit_transform(x)

x = pd.DataFrame(x)

param = {
#       'bagging_freq': 5,
#       'bagging_fraction': 0.32,
        'boost_from_average':'false',
        'boost': 'goss',
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
        'verbosity': 1
    }

num_round = 15000
folds = KFold(n_splits=5, shuffle=False, random_state=32)
oof = np.zeros(len(data))
predictions = np.zeros(len(test))

for fold_no, (trn_idx, val_idx) in enumerate(folds.split(x.values, y.values)):
    print("Fold {}".format(fold_no))
    trn_data = lgb.Dataset(x.iloc[trn_idx], label=y.iloc[trn_idx])
    val_data = lgb.Dataset(x.iloc[val_idx], label=y.iloc[val_idx])
    clf = lgb.train(param, trn_data, num_round, valid_sets = [trn_data, val_data], verbose_eval=500,early_stopping_rounds = 5000)
    oof[val_idx] = clf.predict(x.iloc[val_idx], num_iteration=clf.best_iteration)
    predictions += clf.predict(scaler.transform(test[features]), num_iteration=clf.best_iteration) / folds.n_splits
print("CV score: {:<8.5f}".format(roc_auc_score(y, oof)))

sample_sub = pd.read_csv("C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challenge (Kaggle)/sample_submission.csv")
sample_sub["target"] = predictions

sample_sub.to_csv('C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challenge (Kaggle)/8thsubmission.csv', encoding='utf-8', index=False)