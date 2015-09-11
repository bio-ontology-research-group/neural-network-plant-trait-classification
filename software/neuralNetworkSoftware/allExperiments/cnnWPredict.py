'''
    GPU run command:
        THEANO_FLAGS=mode=FAST_RUN,device=gpu,floatX=float32 python cnnWTest.py
        THEANO_FLAGS='cuda.root=/usr/local/cuda'=mode=FAST_RUN,device=gpu,floatX=float32 python cnnWTest.py
    CPU run command:
'''

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from os.path import expanduser
import os, pandas
import imp, numpy as np

pyvec_api = imp.load_source('api', '../../pyvec/pyvec/core/api.py')

def load_data(directory, custom_height, custom_width, split, withTest):
    print "Loading the data...\n"
<<<<<<< HEAD
    train_data, train_label, val_data, val_label, test_data, test_label, image_list, num_classes = pyvec_api.load_images \
        (directory,custom_height,custom_width, split, withTest)
    return train_data, train_label, val_data, val_label, test_data, test_label, image_list, num_classes


def create_model():
=======
    train_data, train_label, val_data, val_label, test_data, test_label, num_classes = pyvec_api.load_images \
        (directory,custom_height,custom_width, split, withTest)
    return train_data, train_label, val_data, val_label, test_data, test_label, num_classes


def create_model(num_classes):
>>>>>>> 8c5b864afece4f0d6d9d0d388362f11227ff9ab2
    print "Creating the model...\n"
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

    return model

def get_tests(directory):
    test_list =  os.listdir(directory)
    return test_list

<<<<<<< HEAD
def run_experiments(model, train_data, train_label, val_data, val_label, test_data, test_label, batch_size, num_epoch, image_list):
    model.load_weights("./leafForm1.hdf5")
    testY = model.predict_classes(train_data, verbose=0)
    #print "Image:", range(1,len(testY)+1)
    #print "Label:", testY
    #print "images:", image_list
    image_label = dict(zip(image_list,testY))
    return image_label
=======


def run_experiments(model, train_data, train_label, val_data, val_label, test_data, test_label, batch_size, num_epoch):
    model.load_weights("./leafForm.hdf5")
    testY = model.predict_classes(train_data, verbose=0)
    print "Image:", range(1,len(testY)+1)
    print "Label:", testY

>>>>>>> 8c5b864afece4f0d6d9d0d388362f11227ff9ab2

if __name__ == "__main__":
    home_directory = expanduser("~")
    custom_height = 64
    custom_width = 64
<<<<<<< HEAD
    directory = home_directory + "/datasets/predict/herb/"
    test_list = get_tests(directory)
    withTest = True
    split = 1
=======
    directory = home_directory + "/datasets/preProcessed/predict/"
    test_list = get_tests(directory)
    withTest = True
    split = 0.2
>>>>>>> 8c5b864afece4f0d6d9d0d388362f11227ff9ab2
    np.random.seed(1337) # Reproducable results :)
    num_epoch = 20
    batch_size = 128

<<<<<<< HEAD

    for tests in test_list:
        newdir = directory+tests
        print newdir
        print "Trait:", tests
        train_data, train_label, val_data, val_label, test_data, test_label, image_list, num_classes = load_data(newdir, custom_height, custom_width, split, withTest)
	
        model = create_model()
        image_labels = run_experiments(model, train_data, train_label, val_data, val_label, test_data, test_label, batch_size, num_epoch, image_list)

        with open("image_labels1.txt","w") as file1:
            for k,v in image_labels.items():
                file1.write('\n{}: {}'.format(k,v))

=======
    for tests in test_list:
        newdir = directory+tests
        print "Trait:", tests
        train_data, train_label, val_data, val_label, test_data, test_label, num_classes = load_data(newdir, custom_height, custom_width, split, withTest)
        model = create_model(num_classes)
        run_experiments(model, train_data, train_label, val_data, val_label, test_data, test_label, batch_size, num_epoch)
>>>>>>> 8c5b864afece4f0d6d9d0d388362f11227ff9ab2
