'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
        THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu,floatX=float32 python cnn.py
    CPU run command:
        python cnn.py
'''


from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from os.path import expanduser
import os.path
import imp, numpy as np



pyvec_api = imp.load_source('api', '../../pyvec/pyvec/core/api.py')

#batch_size = 100

print "Loading the data...\n"

height = 64
width = 64
directory = '/home/osheak/test'
habm_data = pyvec_api.load_images_unlabelled(directory, height, width)


#print
#print('Train data shape:', train_data.shape)
#print('Validation data shape:', val_data.shape)
#print('Test data shape:', test_data.shape)

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

model.add(Dense(1024, 4))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)

#print "Doing some training and validation..", "with: ", num_epoch, " epochs"
#model.fit(train_data, train_labels, batch_size=batch_size, nb_epoch=num_epoch,
#          show_accuracy=True, verbose=1, validation_data=(val_data, val_labels))

print "testing the model on heb images......\n"
model.load_weights('leafForm.hdf5')
habm_labels = model.predict_classes(habm_data)

print habm_labels
#save weights
model.save_weights('fanf_model.hdf5',overwrite=True);

#print "\nAnd now the test (with", len(test_labels),"samples)..."
#score = model.evaluate(test_data, test_labels, show_accuracy=True, verbose=1, batch_size=batch_size)
#print "Test Accuracy: ", score[1]
