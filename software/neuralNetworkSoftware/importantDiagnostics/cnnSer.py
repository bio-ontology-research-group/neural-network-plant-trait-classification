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
import imp, numpy as np

dataset_getter = imp.load_source('dataset', '../../pyvec/pyvec/input/pickled.py')

home_directory = expanduser("~")

# Data Parameters
custom_height = 64
custom_width = 64
directory = home_directory + "/serialisedData/"
file_name = "importDiagnostics.p"
num_classes = 9
split = 0.7 #Split training and validation (90% for training, 10% validation)

# Training Parameters
np.random.seed(1337) # Reproducable results :)
num_epoch = 20
batch_size = 100

train_data, train_labels, val_data, val_labels, test_data, test_labels = dataset_getter.pickle_in(directory, file_name)
print "Loading the data...\n"


print('Train data shape:', train_data.shape)
print('Validation data shape:', val_data.shape)
print('Test data shape:', test_data.shape)

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

model.add(Dense(1024, num_classes))
model.add(Activation('softmax'))

sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(loss='categorical_crossentropy', optimizer=sgd)


print "Doing some training and validation..", "with: ", num_epoch, " epochs"
model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=num_epoch,
          show_accuracy=True, verbose=1, validation_data=(val_data, val_label))

print "\nAnd now the test (with", len(test_label),"samples)..."
score = model.evaluate(test_data, test_label, show_accuracy=True, verbose=1, batch_size=batch_size)
print "Test Accuracy: ", score[1]
