# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 12:35:14 2019

@author: tanma
"""

import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")

x = data.drop(['ID_code','target'], axis = 1)

y = data["target"]

l = []
for i in range(200):
    l.append(i+1)
    
from sklearn.cross_validation import train_test_split
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest
from sklearn.grid_search import GridSearchCV,RandomizedSearchCV
from sklearn.metrics import f1_score,confusion_matrix

pip_lasso = Pipeline([('scaler',StandardScaler()),('kbest',SelectKBest()),('abc',AdaBoostClassifier())])
param_grid = ({'kbest__k':l})
gbr = GridSearchCV(pip_lasso,param_grid).fit(x_train,y_train)
print(confusion_matrix(gbr.predict(x_test),y_test))