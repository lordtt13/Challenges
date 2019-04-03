# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 17:02:53 2019

@author: tanma
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.ndimage.filters import uniform_filter1d
from sklearn.model_selection import train_test_split

from keras.models import Sequential, Model
from keras.layers import Conv1D, MaxPool1D, Dense, Dropout, Flatten, \
BatchNormalization, Input, concatenate, Activation
from keras.optimizers import Adam

data = pd.read_csv('train.csv')
x = data.drop(['ID_code','target'], axis = 1)
y = data["target"]
y = np.reshape(list(y),(y.shape[0],1)) 

x_train,x_test,y_train,y_test = train_test_split(x,y,test_size = 0.25) 

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)


classifier = Sequential()
classifier.add(Dense(output_dim = 256, init = 'uniform', input_dim = 200,activation = 'relu'))
classifier.add(Dense(output_dim = 64, init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 16, init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 4, init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

def batch_generator(x_train, y_train, batch_size=32):

    half_batch = batch_size // 2
    x_batch = np.empty((batch_size, x_train.shape[1]), dtype='float32')
    y_batch = np.empty((batch_size, 1), dtype='float32')
    
    yes_idx = np.where(y_train[:,0] == 1.)[0]
    non_idx = np.where(y_train[:,0] == 0.)[0]
    
    while True:
        np.random.shuffle(yes_idx)
        np.random.shuffle(non_idx)
    
        x_batch[:half_batch] = x_train[yes_idx[:half_batch]]
        x_batch[half_batch:] = x_train[non_idx[half_batch:batch_size]]
        y_batch[:half_batch] = y_train[yes_idx[:half_batch]]
        y_batch[half_batch:] = y_train[non_idx[half_batch:batch_size]]
    
        for i in range(batch_size):
            sz = np.random.randint(x_batch.shape[1])
            x_batch[i] = np.roll(x_batch[i], sz, axis = 0)
     
        yield x_batch, y_batch

classifier.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics=['accuracy'])
hist = classifier.fit_generator(batch_generator(x_train, y_train, 16), 
                           validation_data = (x_test,y_test), 
                           epochs=40,
                           steps_per_epoch = x_train.shape[0]//16)

plt.plot(hist.history['loss'], color='b')
plt.plot(hist.history['val_loss'], color='r')
plt.show()
plt.plot(hist.history['acc'], color='b')
plt.plot(hist.history['val_acc'], color='r')
plt.show()