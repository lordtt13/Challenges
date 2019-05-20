# -*- coding: utf-8 -*-
"""
Created on Wed Apr  3 13:33:40 2019

@author: tanma
"""

import pandas as pd
import numpy as np


raw_data = pd.read_csv('C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challenge (Kaggle)/train.csv')
x = raw_data.drop(['ID_code','target'],axis = 1)
y = raw_data["target"] 
y = np.reshape(list(y),(y.shape[0],1))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2)

from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

from scipy.ndimage.filters import uniform_filter1d
x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=1)], axis=2)
x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=1)], axis=2)

from keras.models import Model
from keras.layers import CuDNNLSTM, CuDNNGRU, Dropout, Dense, GlobalMaxPool1D, Input, Bidirectional
from keras.layers.merge import concatenate
from keras.callbacks import ModelCheckpoint

inp = Input(shape = x_train.shape[1:])
x = Bidirectional(CuDNNLSTM(units = 256, return_sequences = True))(inp)
x = Dropout(0.5)(x)
x = Bidirectional(CuDNNLSTM(units = 128, return_sequences = True))(x)
x = Dropout(0.5)(x)
x = Bidirectional(CuDNNLSTM(units = 64, return_sequences = True))(x)
x = Dropout(0.5)(x)
x = Bidirectional(CuDNNLSTM(units = 32, return_sequences = True))(x)
y = Bidirectional(CuDNNLSTM(units = 256, return_sequences = True))(inp)
y = Dropout(0.5)(y)
y = Bidirectional(CuDNNLSTM(units = 128, return_sequences = True))(y)
y = Dropout(0.5)(y)
y = Bidirectional(CuDNNLSTM(units = 64, return_sequences = True))(y)
y = Dropout(0.5)(y)
y = Bidirectional(CuDNNLSTM(units = 32, return_sequences = True))(y)
x = concatenate([x,y])
x = GlobalMaxPool1D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu')(x)
x = Dropout(0.25)(x)
x = Dense(64, activation='relu')(x)
x = Dense(1, activation='sigmoid')(x)

model = Model(inp,x)

def batch_generator(x_train, y_train, batch_size=32):

    half_batch = batch_size // 2
    x_batch = np.empty((batch_size, x_train.shape[1], x_train.shape[2]), dtype='float32')
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

filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath,monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callbacks=[checkpoint]
        
model.compile(optimizer='nadam', loss = 'binary_crossentropy', metrics=['accuracy'])
hist = model.fit(x_train, y_train,
                           validation_data = (x_test,y_test), 
                           epochs = 20,
                           batch_size = 256,callbacks = callbacks)


test = pd.read_csv("C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challenge (Kaggle)/test.csv")
sample_sub = pd.read_csv("C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challenge (Kaggle)/sample_submission.csv")
test = StandardScaler().fit_transform(test.drop('ID_code',axis = 1))
test = np.stack([test, uniform_filter1d(test, axis=1, size=1)], axis=2)
sample_sub["target"] = model.predict(test)
