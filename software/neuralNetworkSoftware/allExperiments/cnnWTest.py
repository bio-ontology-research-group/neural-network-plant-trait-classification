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
from sklearn.metrics import confusion_matrix
import os
import imp, numpy as np

pyvec_api = imp.load_source('api', '../../pyvec/pyvec/core/api.py')

def load_data(directory, custom_height, custom_width, split, withTest):
    print "Loading the data...\n"
    train_data, train_label, val_data, val_label, test_data, test_label, image_list, num_classes = pyvec_api.load_images \
        (directory,custom_height,custom_width, split, withTest)
    return train_data, train_label, val_data, val_label, test_data, test_label, image_list, num_classes
    train_data, train_label, val_data, val_label, test_data, test_label, num_classes = pyvec_api.load_images \
        (directory,custom_height,custom_width, split, withTest)
    return train_data, train_label, val_data, val_label, test_data, test_label, num_classes


def create_model(num_classes):
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

    model.add(Dense(1024, num_classes))
    model.add(Activation('softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd)

    return model

def get_tests(directory):
    test_list =  os.listdir(directory)
    return test_list

def run_experiments(model, train_data, train_label, val_data, val_label, test_data, test_label, batch_size, num_epoch):
    print "Doing some training and validation..", "with: ", num_epoch, " epochs"
    model.fit(train_data, train_label, batch_size=batch_size, nb_epoch=num_epoch,
             show_accuracy=True, verbose=1, validation_data=(val_data, val_label))

    print "\nAnd now the test (with", len(test_label),"samples)..."
    score = model.evaluate(test_data, test_label, show_accuracy=True, verbose=1, batch_size=batch_size)
    print "Test Accuracy: ", score[1]
    Ytest = np.nonzero(test_label)
    Ytest = Ytest[1].tolist() 
    YPredict = model.predict_classes(test_data, verbose=0)   
    YPredict = YPredict.tolist()
    confusion_mat = confusion_matrix(Ytest, YPredict)
    print "confusion matrix: \n",confusion_mat
    model.save_weights("leafStructure.hdf5", overwrite=True)
    
    
    # run tests on herb 
def test_model(model, data, label, image_list):
    testY = model.predict_classes(data, verbose=0)
    image_label = dict(zip(image_list,testY))
    return image_label

   
if __name__ == "__main__":
    home_directory = expanduser("~")
    custom_height = 64
    custom_width = 64
    directory = home_directory + "/datasets/preProcessed1/"
    test_list = get_tests(directory)
    withTest = True
    split = 0.7
    np.random.seed(1337) # Reproducable results :)
    num_epoch = 20
    batch_size = 128

    for tests in test_list:
        newdir = directory+tests
        print "Trait:", tests
        train_data, train_label, val_data, val_label, test_data, test_label, _, num_classes = load_data(newdir, custom_height, custom_width, split, withTest)
        model = create_model(num_classes)
        run_experiments(model, train_data, train_label, val_data, val_label, test_data, test_label, batch_size, num_epoch)

  
    # apply learned model on herb unlabelled data 
 
    newdir = home_directory + "/datasets/predict/herb/"
    test_list = get_tests(newdir)
    split = 1 #no split

    for tests in test_list:
        newdir = newdir +tests
        print "Trait:", tests
        data, label, _, _, _, _, image_list, _ = load_data(newdir, custom_height, custom_width, split, withTest)
	image_labels = test_model(model, data, label, image_list)

        with open("leafStr_labels.txt","w") as file1:
	    for k,v in image_labels.items():
                file1.write('\n{}: {}'.format(k,v))
