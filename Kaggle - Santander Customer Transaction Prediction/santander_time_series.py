# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 00:33:45 2019

@author: tanma
"""
import pandas as pd
import numpy as np


raw_data = pd.read_csv('C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challenge (Kaggle)/train.csv')
x = raw_data.drop(['ID_code','target'],axis = 1)
y = raw_data["target"] 
y = np.reshape(list(y),(y.shape[0],1))

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.25)

from sklearn.preprocessing import StandardScaler
x_train = StandardScaler().fit_transform(x_train)
x_test = StandardScaler().fit_transform(x_test)

from scipy.ndimage.filters import uniform_filter1d
x_train = np.stack([x_train, uniform_filter1d(x_train, axis=1, size=1)], axis=2)
x_test = np.stack([x_test, uniform_filter1d(x_test, axis=1, size=1)], axis=2)

from keras.models import Sequential
from keras.layers import Conv1D, MaxPool1D, BatchNormalization, Dropout, Dense, GlobalMaxPool1D
from keras.callbacks import ModelCheckpoint

model = Sequential()
model.add(Conv1D(filters=8, kernel_size=11, activation='relu', input_shape=x_train.shape[1:]))
model.add(MaxPool1D(strides=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=16, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=32, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=2))
model.add(BatchNormalization())
model.add(Conv1D(filters=64, kernel_size=11, activation='relu'))
model.add(MaxPool1D(strides=2))
model.add(GlobalMaxPool1D())
model.add(Dropout(0.5))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.25))
model.add(Dense(64, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

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
hist = model.fit_generator(batch_generator(x_train, y_train, 32), 
                           validation_data = (x_test,y_test), 
                           epochs = 40,
                           steps_per_epoch = 1000,callbacks = callbacks)

features = x.columns
test = pd.read_csv("C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challenge (Kaggle)/test.csv")
sample_sub = pd.read_csv("C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challenge (Kaggle)/sample_submission.csv")
test = test[features]
test = StandardScaler().fit_transform(test)
test = np.stack([test, uniform_filter1d(test, axis=1, size=7)], axis=2)
sample_sub["target"] = model.predict(test.drop('ID_code',axis = 1))

sample_sub.to_csv('C:\\Users\\tanma.TANMAY-STATION\\Desktop\\Santander Challenge (Kaggle)/8thsubmission.csv', encoding='utf-8', index=False)