# -*- coding: utf-8 -*-
"""
Created on Mon Mar 11 22:32:53 2019

@author: tanma
"""

import pandas as pd
import numpy as np

data = pd.read_csv("train.csv")

x = data.drop(['ID_code','target'], axis = 1)
y = data['target']
features = x.columns

from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier
rand_clf = RandomForestClassifier()
rand_clf.fit(data.iloc[:,2:],data.iloc[:,1])
importance = rand_clf.feature_importances_

df = pd.DataFrame({'importance_q': importance.flatten(), 'Label': features})
df = df.sort_values(by=['importance_q'],ascending = False)
important_vars = df["Label"][df.importance_q > 0.005].tolist()
    
x = data[important_vars]


from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test = train_test_split(x,y,test_size = 0.2)

from keras.models import Sequential,Model
from keras.layers import Dense,Input,Dropout
from keras.callbacks import ModelCheckpoint
from keras.layers.advanced_activations import LeakyReLU
from keras import backend as K
from keras.callbacks import Callback
from sklearn.metrics import accuracy_score,roc_auc_score

classifier = Sequential()
classifier.add(Dense(output_dim = 256, init = 'uniform', input_dim = len(important_vars),activation = 'relu'))
classifier.add(Dense(output_dim = 64, init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 16, init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 4, init = 'uniform',activation = 'relu'))
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))


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

class roc_callback(Callback):
    def __init__(self,training_data,validation_data):
        self.x = training_data[0]
        self.y = training_data[1]
        self.x_val = validation_data[0]
        self.y_val = validation_data[1]


    def on_train_begin(self, logs={}):
        return

    def on_train_end(self, logs={}):
        return

    def on_epoch_begin(self, epoch, logs={}):
        return

    def on_epoch_end(self, epoch, logs={}):
        y_pred = self.model.predict(self.x)
        roc = roc_auc_score(self.y, y_pred)
        y_pred_val = self.model.predict(self.x_val)
        roc_val = roc_auc_score(self.y_val, y_pred_val)
        print('\rroc-auc: %s - roc-auc_val: %s' % (str(round(roc,4)),str(round(roc_val,4))),end=100*' '+'\n')
        return

    def on_batch_begin(self, batch, logs={}):
        return

    def on_batch_end(self, batch, logs={}):
        return
    
classifier.compile(loss='binary_crossentropy',
          optimizer= "nadam", metrics = [f1])

from keras.callbacks import EarlyStopping
filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
earlystopping = EarlyStopping(patience = 2)
checkpoint = ModelCheckpoint(filepath,monitor='val_f1',verbose=1,save_best_only=True,mode='max')
callbacks=[checkpoint,roc_callback(training_data=(X_train, y_train),validation_data=(X_test, y_test)),earlystopping]


i = 32
while i<1024:
    classifier.fit(x,y,epochs = 10,batch_size = i,callbacks = callbacks,validation_split = 0.25)
    i = i + 32

test_file = pd.read_csv("test.csv")
x_test = test_file[important_vars]
test_pred = classifier.predict(x_test)

sample_sub = pd.read_csv("sample_submission.csv")
sample_sub["target"] = test_pred

sample_sub.to_csv('1stsubmission.csv', encoding='utf-8', index=False)