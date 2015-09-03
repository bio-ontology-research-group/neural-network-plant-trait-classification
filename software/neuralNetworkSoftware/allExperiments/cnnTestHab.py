from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from os.path import expanduser
import os.path
import imp, numpy as np

pyvec_api = imp.load_source('api', '../../pyvec/pyvec/core/api.py')

print "Loading the data...\n"

height = 64
width = 64
directory = '/home/alshahmm/datasets/PicturaE_Hachmann'
habm_data = pyvec_api.load_images_unlabelled(directory, height, width)
print len(habm_data)
print "\nCreating the model...\n"

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
model.add(Flatten())

model.add(Dense(128*16*16, 1024))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(1024, 5))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


print "testing the model on heb images......\n"
model.load_weights('leafForm.hdf5')
habm_labels = model.predict_classes(habm_data)

print habm_labels
