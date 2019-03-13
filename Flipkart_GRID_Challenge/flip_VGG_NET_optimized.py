# -- coding: utf-8 --
"""
Created on Sun Feb  3 00:24:20 2019

@author: tanma
"""

import cv2
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

folder = "F:\FLIPKART GRID Challenge Images/images"

train_file = pd.read_csv("F:\FLIPKART GRID Challenge Images/training.csv")
train_file_names = train_file['image_name'] [:14000]

image_arr = []
for i in train_file_names.values:
    path = os.path.join(folder,i)
    image = cv2.imread(path)
    image = cv2.resize(image,(64,64))
    image_arr.append(image)
dict_data = {"training_example" : image_arr}

with open ("image_arr1.pkl","wb") as file:
    image_arr = pickle.dump(image_arr,file)
    
from keras.layers import Input,Dense,Dropout,Convolution2D,Flatten,MaxPooling2D,GlobalMaxPooling2D
from keras.models import Model,Sequential
from keras.activations import relu,sigmoid,softmax
from keras.layers import BatchNormalization,GlobalAveragePooling2D
from keras import optimizers
from keras.callbacks import ModelCheckpoint,EarlyStopping,TensorBoard
from keras.regularizers import l2,l1
from keras.utils import plot_model
from keras.layers.advanced_activations import LeakyReLU
import keras.backend as K
import tensorflow as tf

model = Sequential()
model.add(Convolution2D(128, (3, 3), input_shape = (64, 64, 3)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Convolution2D(64, (3, 3)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(Convolution2D(64, (3, 3)))
model.add(LeakyReLU())
model.add(BatchNormalization())
model.add(Convolution2D(64, (3, 3)))
model.add(LeakyReLU())
model.add(MaxPooling2D(pool_size = (2, 2)))
model.add(GlobalAveragePooling2D())
model.add(Dense(units = 256))
model.add(LeakyReLU())
model.add(Dropout(0.2))
model.add(Dense(units = 4))
model.add(LeakyReLU())
model.compile(optimizer = 'nadam', loss = 'mse')
filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
early_stopping = EarlyStopping(monitor='loss')
callbacks=[checkpoint,early_stopping]


with open("image_arr1.pkl","rb") as file:
    input_train = pickle.load(file)
    
y = train_file.iloc[:,1:5].values

model.fit(np.array(input_train).reshape(14000,64,64,3),y,epochs = 20,batch_size = 32,callbacks=callbacks) 

pred = model.predict(np.array(input_train).reshape(14000, 64, 64, 3))

for i in range(len(pred)):
    print(y[i],np.rint(pred[i]))
    
# Important Codes to Remember:
"""
# To create a pickle Dump:
with open ("image_arr1.pkl","wb") as file:
    image_arr = pickle.dump(image_arr,file)
    
# To read files:
with open("image_arr1.pkl","rb") as file:
    input_train = pickle.load(file)

# To throttle GPU Resource Allocation:
import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.7
set_session(tf.Session(config=config))
"""