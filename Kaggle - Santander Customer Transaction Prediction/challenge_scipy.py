# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 13:09:46 2019

@author: tanma
"""
import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")

X = data.drop(['ID_code','target'], axis = 1)

y = data["target"]

import statsmodels.formula.api as sm
X = np.append(arr = np.ones((200000, 1)).astype(int), values = X, axis = 1)

l = []
i = 0
while len(l) < 201:
    l.append(i)
    i = i + 1

X_opt = X[:,l]
regressor_OLS = sm.OLS(endog = y, exog = X_opt).fit()
regressor_OLS.summary()