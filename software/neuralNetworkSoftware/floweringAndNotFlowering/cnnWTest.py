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
from keras.optimizers import *
from keras.utils import np_utils, generic_utils
from os.path import expanduser
import  imp, numpy as np

dataset_getter = imp.load_source('dataset', '../../pyvec/pyvec/images/dataset.py')

home_directory = expanduser("~")

# Data Parameters
custom_height = 64
custom_width = 64
directory = home_directory + "/datasets/fanf/preProcessed/"
num_classes = 20
split = 0.9 #Split training and validation (90% for training, 10% validation)

# Training Parameters
np.random.seed(1337) # Reproducable results :)
num_epoch = 2
batch_size = 100


print "Loading the data...\n"

# I have already preprocessed the data, if you haven't done this please look at the examples
# provided with pyvec: https://github.com/KeironO/Pyvec/blob/master/examples/loadingImages/loading_images.py
X_train, Y_train, X_val, Y_val, X_test, Y_test = dataset_getter.vectorise(directory,num_classes,custom_height,
                                                                custom_width, split, True)
y_train = np_utils.to_categorical(Y_train, num_classes)
y_val = np_utils.to_categorical(Y_val, num_classes)
y_test = np_utils.to_categorical(Y_test, num_classes)

nb_train = len(Y_train)
nb_validation = len(Y_val)
nb_test = len (Y_test)
print "Images for training: ", nb_train,  "\n", "Images for validation: ", nb_validation, "\n" + "Images for testing: ",  nb_test, "\n"

print "Creating the model...\n"

model = Sequential()

model.add(Convolution2D(custom_height, 3, 3, 3, border_mode='full'))
model.add(Activation('relu'))
model.add(Convolution2D(custom_height, custom_width, 3, 3))
model.add(Activation('relu'))
model.add(MaxPooling2D(poolsize=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(65536, 128))
model.add(Activation('relu'))
model.add(Dropout(0.5))

model.add(Dense(128, num_classes))
model.add(Activation('softmax'))

rms = Adadelta()
model.compile(loss='categorical_crossentropy', optimizer=rms)

best_validation_accuracy = 0.0

print "Training on", num_epoch, "epoch/s..."

for e in range(num_epoch):
    print "\nEpoch", e
    print "Training..."
    batch_num = len(y_train)/batch_size
    progbar = generic_utils.Progbar(X_train.shape[0])
    for i in range(batch_num):
        train_loss,train_accuracy = model.train_on_batch(X_train[i*batch_size:(i+1)*batch_size], Y_train[i*batch_size:(i+1)*batch_size], accuracy=True)
        progbar.add(batch_size, values=[("Current TL:", train_loss), ("Current A:", train_accuracy)] )
    print "\nRunning a little validation..."
    val_loss,val_accuracy = model.evaluate(X_val, y_val, batch_size=100,show_accuracy=True)

    if best_validation_accuracy < val_accuracy:
        best_validation_accuracyaccuracy = val_accuracy

print "\nBest Validation Accuracy: ", best_validation_accuracy

print "\nTest time!"
test_loss,test_accuracy = model.evaluate(X_test, y_test, batch_size=1,show_accuracy=True)
print "\n Final Test Accuracy:",test_accuracy

