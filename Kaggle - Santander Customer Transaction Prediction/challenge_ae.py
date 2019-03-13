# -*- coding: utf-8 -*-
"""
Created on Wed Mar 13 01:29:59 2019

@author: tanma
"""

import pandas as pd
import numpy as np

import tensorflow as tf
import keras
from keras.models import Model, load_model
from keras.layers import Input, Dense
from keras.callbacks import ModelCheckpoint, TensorBoard
from keras import regularizers 
from sklearn.preprocessing import  StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from IPython.display import display, Math, Latex
import keras.backend as K

data = pd.read_csv("train.csv")

X = data.drop(['ID_code','target'], axis = 1)

y = data["target"]

def f1(y_true, y_pred):
    def recall(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):

        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2*((precision*recall)/(precision+recall+K.epsilon()))

X_train , X_test, y_train,y_test = train_test_split(X ,y ,test_size = 0.2)

scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled  = scaler.transform(X_test)

input_dim = X_train.shape[1]
encoding_dim = 6

input_layer = Input(shape=(input_dim, ))
encoder = Dense(encoding_dim, activation="tanh",activity_regularizer=regularizers.l1(10e-5))(input_layer)
encoder = Dense(int(encoding_dim / 2), activation="tanh")(encoder)
encoder = Dense(int(2), activation="tanh")(encoder)
decoder = Dense(int(encoding_dim/ 2), activation='tanh')(encoder)
decoder = Dense(int(encoding_dim), activation='tanh')(decoder)
decoder = Dense(input_dim, activation='tanh')(decoder)
autoencoder = Model(inputs=input_layer, outputs=decoder)

nb_epoch = 100
batch_size = 50
autoencoder.compile(optimizer='nadam', loss='mse')

filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath,monitor='val_loss',verbose=1,save_best_only=True,mode='min')
callbacks=[checkpoint]

autoencoder.fit(X_train_scaled, X_train_scaled,
                        epochs=nb_epoch,
                        batch_size=batch_size,
                        shuffle=True,
                        validation_split=0.1,
                        verbose=0,
                        callbacks = callbacks
                        )


test_file = pd.read_csv("test.csv")
x_test = scaler.transform(test_file.drop(['ID_code'], axis = 1))
test_pred = autoencoder.predict(x_test)

sample_sub = pd.read_csv("sample_submission.csv")

mse = np.mean(np.power(x_test - test_pred, 2), axis=1)
df_error = pd.DataFrame({'reconstruction_error': mse, 'Label': sample_sub['ID_code']}, index=sample_sub.index)

outliers = df_error['Label'][df_error.reconstruction_error > np.mean(mse)].tolist()

for i,j in enumerate(df_error['reconstruction_error']):
    if j > np.mean(mse):
        sample_sub['target'].values[i] = 1
    else:
        sample_sub['target'].values[i] = 0

sample_sub.to_csv('3rdsubmission.csv', encoding='utf-8', index=False)   