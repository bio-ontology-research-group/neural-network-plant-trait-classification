#*-*coding:utf8*-*#
import os
from PIL import Image
import numpy as np

def crop(img):
	imgarray = np.asarray(img,dtype="uint8")
	arr0 = imgarray[:,:,0]+imgarray[:,:,1]+imgarray[:,:,2]
	sum_column = arr0.sum(axis=0)
	i,j = 0,len(sum_column)-1
	start_column,end_column = i,j
	while(sum_column[i] < sum_column.min()+(sum_column.max()-sum_column.min())/50):
		start_column = i
		i += 1
	while(sum_column[j]< sum_column.min()+(sum_column.max()-sum_column.min())/50):
		end_column = j
		j -= 1
	sum_row = arr0.sum(axis=1)
	i,j = 0,len(sum_row)-1
	start_row,end_row = i,j
	while(sum_row[i]< sum_row.min()+(sum_row.max()-sum_row.min())/50):
		start_row = i
		i += 1
	while(sum_row[j]< sum_row.min()+(sum_row.max()-sum_row.min())/50):
		end_row = j
		j -= 1
	newarray = imgarray[start_row:end_row,start_column:end_column,:]
	img = Image.fromarray(newarray,"RGB")
	return img


def tosquare(img):
	width,height = img.size
	if height<width:
		black_len = (width - height)/2
		imgarray = np.asarray(img)
		newarray = np.zeros((64,64,3),dtype="uint8")
		newarray[black_len:black_len+height,:,:]=imgarray[:,:,:]
		img = Image.fromarray(newarray,"RGB")
	if height>width:
		l = height - width
		imgarray = np.asarray(img)
		img = Image.fromarray(imgarray[0:height-l,:,:],"RGB")
	return img


if __name__ == "__main__":
    directory = "/home/osheak/datasets/fanfMid/imageFiles/numberedFolders/"
    save_path = "/home/osheak/datasets/fanfMid/imageFiles/numberedFoldersTest/"
    if not os.path.exists(save_path):
        os.makedirs("/home/osheak/datasets/fanfMid/imageFiles/numberedFoldersTest/")
	for root, dirs, files in os.walk(directory):
		for name in files:
			if name.endswith((".jpg", ".JPG")):
				print name
    # for i in xrange(len(imglist)):
    #     imgname = imglist[i]
    #     print imgname
    #     img = Image.open(directory+"/"+imgname)
    #     img = crop(img)
    #     width,height = img.size
    #     img = img.resize((64,64*height/width))
    #     img = tosquare(img)
    #     img.save(save_path  +imgname)