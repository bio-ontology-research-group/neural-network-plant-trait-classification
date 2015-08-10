__author__ = 'osheak'

import os
from PIL import Image
import numpy as np
import csv


def get_trainlabel():
    table = {}
    with open("./fanfLabels.csv", "rb") as traitsFile:
        traitsReader = csv.reader(traitsFile, delimiter="\t", lineterminator="\n")
        traitsReader.next()

        for traits in traitsReader:
            table[traits[0]] = int(traits[1])
        return table

def load_colour_images(directory="/home/osheak/datasets/fanfMid/imageFiles/test_RGB_Plus/"):
    imgs = os.listdir(directory)
    num = len(imgs)
    table = get_trainlabel()
    traindata = np.empty((num,3,28,28),dtype="float32")
    traindata.flatten()
    trainlabel = np.empty((num,),dtype="uint8")
    for i in range(num):
        imgname = imgs[i]
        img = Image.open(directory+imgname)
        arr = np.asarray(img, dtype="float32")
        traindata[i,:,:,:] = [arr[:,:,0],arr[:,:,1],arr[:,:,2]]
        trainlabel[i] = table[imgname.split('.')[0]]
    return traindata,trainlabel


