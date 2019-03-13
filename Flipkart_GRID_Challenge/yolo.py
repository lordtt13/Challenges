# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 18:30:33 2019

@author: tanma
"""

from keras.models import Sequential, Model
from keras.layers import Input, BatchNormalization, Flatten, Dense, Reshape
from keras.layers import Activation, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from keras.applications.vgg19 import VGG19
from keras import backend as K

def vgg_yolo():
    vgg19 = VGG19(include_top=False, weights='imagenet',
                  input_tensor=None, input_shape=(64,64,3))
    ip = Input(shape=(64, 64, 3))

    h = vgg19.layers[1](ip)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.1)(h)
    h = vgg19.layers[2](h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)

    for i in range(4, 6):
        h = vgg19.layers[i](h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)

    for i in range(7, 11):
        h = vgg19.layers[i](h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)
    
    h = Flatten()(h)
    h = Dense(units = 64)(h)
    h = LeakyReLU(alpha=0.1)(h)
    h = Dropout(0.2)(h)
    h = Dense(units = 4)(h)

    return Model(ip, h)

def yolo():
    ip = Input(shape=(64, 64, 3))
    h = Conv2D(16, (3, 3), strides=(1, 1), padding='same', use_bias=False)(ip)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2))(h)

    for i in range(0, 4):
        h = Conv2D(32*(2**i), (3, 3), strides=(1, 1), padding='same', use_bias=False)(h)
        h = BatchNormalization()(h)
        h = LeakyReLU(alpha=0.1)(h)
        h = MaxPool2D(pool_size=(2, 2))(h)

    h = Conv2D(512, (3, 3), strides=(1, 1), padding='same', use_bias=False)(h)
    h = BatchNormalization()(h)
    h = LeakyReLU(alpha=0.1)(h)
    h = MaxPool2D(pool_size=(2, 2), strides=(1, 1), padding='same')(h)

    
    h = Flatten()(h)
    h = Dense(units = 64)(h)
    h = LeakyReLU(alpha=0.1)(h)
    h = Dropout(0.2)(h)
    h = Dense(units = 4)(h)


    return Model(ip, h)

