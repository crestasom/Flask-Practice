# -*- coding: utf-8 -*-
"""
Created on Fri May 18 09:00:25 2018

@author: Public
"""

from keras.models import model_from_json
import tensorflow as tf

import keras
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers.core import Activation, Flatten, Dense, Dropout
from keras.layers.convolutional import Conv2D
from keras.layers.pooling import MaxPool2D
#from keras.layers.convolutional import Convolution2D, MaxPooling2D
from sklearn.model_selection import train_test_split

def init():
  #json_file = open('model.json','r')
 # loaded_model_json = json_file.read()
  #json_file.close()
  #loaded_model = model_from_json(loaded_model_json)
  #load weights into new model
  loaded_model=LeNet(28, 28, 1, 10)
  loaded_model.load_weights("F:/flasktut/model/model.h5")
  print("Loaded Model from disk")
  #compile and evaluate loaded model
  loaded_model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
  graph = tf.get_default_graph()
  return loaded_model,graph


def LeNet(width, height, channels, output):
    model = Sequential()
    
    #Convulation
    model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(2,2), input_shape=(width, height, channels)))
    
    #ReLU Activation
    model.add(Activation('relu'))
    
    #Pooling
    model.add(MaxPool2D(pool_size=(2,2)))
    
    #Convolution
    model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(2,2)))
    
    #ReLU Activation
    model.add(Activation('relu'))
    
    #Pooling
    model.add(MaxPool2D(pool_size=(2,2)))
    
    model.add(Flatten())
    
    #Hidden Layer
    model.add(Dense(100))
    
    model.add(Activation('relu'))
    
    model.add(Dense(output))
    
    model.add(Activation('softmax'))
    
    return model
    
    #We can also add dropout
    
