# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 18:23:41 2019

@author: tanma
"""


import cv2
from PIL import Image
import pickle
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt

folder = "D:\FLIPKART GRID Challenge Images/images"

train_file = pd.read_csv("D:\FLIPKART GRID Challenge Images/training.csv")
train_file_names = train_file['image_name'] 

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
from yolo import yolo,vgg_yolo

# model = vgg_yolo()
model = yolo()

model.compile(optimizer = 'nadam', loss = 'mse')
filepath = "weight-improvement-{epoch:02d}-{loss:4f}.hd5"
checkpoint = ModelCheckpoint(filepath,monitor='loss',verbose=1,save_best_only=True,mode='min')
callbacks=[checkpoint]


with open("image_arr1.pkl","rb") as file:
    input_train = pickle.load(file)
    
y = train_file.iloc[:,1:5].values

i = 64
while i<14000:
    model.fit(np.array(input_train).reshape(14000,64,64,3),y,epochs = 10,batch_size = i,callbacks = callbacks)
    i = i + 32

"""
pred = model.predict(np.array(input_train).reshape(14000, 64, 64, 3))

for i in range(len(pred)):
    print(y[i],np.rint(pred[i]))
    
# Important Codes to Remember:

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
config.gpu_options.per_process_gpu_memory_fraction = 0.6
set_session(tf.Session(config=config))
"""

test_file = pd.read_csv("D:\\FLIPKART GRID Challenge Images\\test.csv")
test_file_names = test_file['image_name'] 

image_arr_test = []
for i in test_file_names.values:
    path = os.path.join(folder,i)
    image1 = cv2.imread(path)
    image1 = cv2.resize(image1,(64,64))
    image_arr_test.append(image1)
dict_data = {"test_example" : image_arr_test}

with open ("image_arr2.pkl","wb") as file:
    image_arr_test = pickle.dump(image_arr_test,file)
    
with open("image_arr2.pkl","rb") as file:
    input_test = pickle.load(file)
    
test_pred = model.predict(np.array(input_test).reshape(12815, 64, 64, 3))
    
test_file['x1'] = np.rint(test_pred[:,0].reshape(-1,1)) 
test_file['x2'] = np.rint(test_pred[:,1].reshape(-1,1)) 
test_file['y1'] = np.rint(test_pred[:,2].reshape(-1,1)) 
test_file['y2'] = np.rint(test_pred[:,3].reshape(-1,1))

test_file.to_csv('test5.csv', encoding='utf-8', index=False)
