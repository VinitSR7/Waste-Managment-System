import numpy as np
from numpy import array
import pandas as pd
import matplotlib.pyplot as plt 
import string
import os
from PIL import Image
#import glob
import keras
import tensorflow as tf
from pickle import dump, load
from time import time
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Embedding, TimeDistributed, Dense, RepeatVector,\
                         Activation, Flatten, Reshape, concatenate, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from keras.layers.wrappers import Bidirectional
from keras.layers.merge import add
from tensorflow.keras.applications.inception_v3 import InceptionV3

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras import Input, layers
from tensorflow.keras import optimizers
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def define_model():
  model = Sequential()
  model.add(Dense(256, activation='relu', input_dim = 2048))
  model.add(Dropout(0.4))
  model.add(Dense(256, activation = 'relu'))
  model.add(Dropout(0.3))
  model.add(Dense(1, activation='sigmoid'))
  return model

model = define_model()

model.compile(optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy'])

model.load_weights('static/models/my_model.h5')



model_new = InceptionV3()
model_new = Model(model_new.input, model_new.layers[-2].output)
model_new.load_weights('static/models/inception.h5')

def preprocess(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    
    x = image.img_to_array(img)

    x = np.expand_dims(x, axis=0)
    
    x = preprocess_input(x)
    return x

def encode(image):
    image = preprocess(image) 
    v = model_new.predict(image) 
    v = np.reshape(v, v.shape[1]) 
    return v

def make_prediction(filename):
  #x = plt.imread(filename)
  #plt.imshow(x)
  v = encode(filename)
  # v = v.astype('float32')
  # v /= 255
  pred =  model.predict(v.reshape(1, v.shape[0]))
  if(pred[0] == 1):
    return "Garbage found"
  return "NO Garbage"

if __name__ == "__main__":
    print("OK")
   # print(predict("garbage.jpeg"))
