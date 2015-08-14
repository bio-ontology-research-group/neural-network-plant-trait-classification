__author__ = 'osheak'

import os
from PIL import Image
import numpy as np
import random
from keras.utils import np_utils, generic_utils

def load_directories(directory):
	imglist = []
	for dirname,dirnames,filenames in os.walk(directory):
		for filename in filenames:
			label = os.path.basename(os.path.normpath(dirname))
			imglist.append([label, filename])
	return imglist

def load_colour_images(directory):
    imglist = load_directories(directory)
    num = len(imglist)
    traindata = np.empty((num, 3, 64, 64), dtype="float32")
    traindata.flatten()
    trainlabel = np.empty((num,),dtype="uint8")
    for i, imgname in enumerate(imglist):
        img = Image.open(directory+"/"+imgname[0]+"/"+imgname[1])
        arr = np.asarray(img, dtype="float32")
        traindata[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
        trainlabel[i] = imgname[0]
    return traindata, trainlabel

def vectorise(directory, nb_classes, height, width):
    data, label0 = load_colour_images(directory)
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

    X_train = X_train.reshape(X_train.shape[0], 3, height, width)/255
    X_val = X_val.reshape(X_val.shape[0], 3, height, width)/255
    X_train = X_train.astype("float32")
    X_val = X_val.astype("float32")

    print('X_train shape:', X_train.shape)
    print('Y_train shape:', Y_train.shape)

    return X_train, Y_train, X_val, Y_val
