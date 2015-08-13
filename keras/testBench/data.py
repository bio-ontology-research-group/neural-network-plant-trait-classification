__author__ = 'osheak'

import os
from PIL import Image
import numpy as np
import csv

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
