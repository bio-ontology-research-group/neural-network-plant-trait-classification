#coding:utf-8

'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
        THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
'''

from __future__ import absolute_import
from __future__ import print_function
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.advanced_activations import PReLU
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD, Adadelta, Adagrad
from keras.utils import np_utils, generic_utils
from six.moves import *
import random,cPickle
from data import load_colour_images
import numpy as np


np.random.seed(1993)
nb_epoch = 20
batch_size = 100
nb_classes = 2

data, label0 = load_colour_images("/home/osheak/datasets/fanfMid/imageFiles/test_RGB_plus/")
num = len(label0)
index = [i for i in range(num)]
random.shuffle(index)
data = data[index]
label0 = label0[index]

label = np_utils.to_categorical(label0, nb_classes)

X_train =  data[0: 14850]
Y_train = label[0 : 14850]

X_val = data[0: 1648]
Y_val = label[0: 1648]

X_train = X_train.reshape(X_train.shape[0], 3, 64, 64)/255
X_val = X_val.reshape(X_val.shape[0], 3, 64, 64)/255
X_train = X_train.astype("float32")
X_val = X_val.astype("float32")

print('X_train shape:', X_train.shape)
print('Y_train shape:', Y_train.shape)

y_train = np_utils.to_categorical(Y_train, nb_classes)
y_val = np_utils.to_categorical(Y_val, nb_classes)

model = Sequential()

model.add(Convolution2D(64, 3, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(64, 64, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 64, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Convolution2D(128, 128, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(128, 128, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(8192, 100))
model.add(Activation('relu'))
model.add(Dropout(0.5))




model.add(Dense(100, nb_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='mse', optimizer=sgd)


nb_train = len(Y_train)
nb_validation = len(Y_val)
print( 'train samples:',nb_train, 'validation samples:',nb_validation)


model.fit(X_train, y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, verbose=1, validation_data=(X_val, y_val))
score = model.evaluate(X_train, y_train, show_accuracy=True, verbose=0)
print (score[0])
