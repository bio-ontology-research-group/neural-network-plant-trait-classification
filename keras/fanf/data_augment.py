import os, shutil
import cv2
import csv

def get_trainlabel():
    table = {}
    with open("./fanfLabels.csv", "rb") as traitsFile:
        traitsReader = csv.reader(traitsFile, delimiter="\t", lineterminator="\n")
        traitsReader.next()

        for traits in traitsReader:
            table[traits[0]] = int(traits[1])
        return table

table = get_trainlabel()
srcdir = "/home/osheak/datasets/fanfMid/imageFiles/test_RGB/"
newdir = "/home/osheak/datasets/fanfMid/imageFiles/test_RGB_plus/"
if not os.path.exists(newdir):
    os.makedirs(newdir)
imgs = os.listdir(srcdir)

for imgname in imgs:
        label = table[imgname.split('.')[0]]
        if label == 1:
            img = cv2.imread(srcdir+imgname)
            cv2.imwrite(newdir+imgname.split('.')[0]+'.1.jpg',img)
        elif label == 0:
            img = cv2.imread(srcdir+imgname)
            cv2.imwrite(newdir+imgname.split('.')[0]+'.0.jpg',img)
