import os, cv2

if __name__=="__main__":
    image_height = 64
    image_width = 64
    directory = "/home/osheak/datasets/preProcessed/lVena"
    image_list = []
    print os.walk(directory)
    for dir_name, dir_names ,file_names in os.walk(directory):
        for file in file_names:
            label = os.path.basename(os.path.normpath(dir_name))
            image_list.append([label, file])
    rotation_matrix_1 = cv2.getRotationMatrix2D((image_height/2,image_width/2), 4, 1)
    rotation_matrix_2 = cv2.getRotationMatrix2D((image_height/2,image_width/2), 7, 1)
    rotation_matrix_3 = cv2.getRotationMatrix2D((image_height/2,image_width/2), -7, 1)
    rotation_matrix_4 = cv2.getRotationMatrix2D((image_height/2,image_width/2), -4, 1)
    #I want this to go into memory.
    # for images in image_list:
        # image = cv2.imread(directory+"/"+images[0]+"/"+images[1])
        # rotated_image_1 = cv2.warpAffine(image, rotation_matrix_1, (image_height, image_width))
        # rotated_image_2 = cv2.warpAffine(image, rotation_matrix_2, (image_height, image_width))
        # rotated_image_3 = cv2.warpAffine(image, rotation_matrix_3, (image_height, image_width))
        # rotated_image_4 = cv2.warpAffine(image, rotation_matrix_4, (image_height, image_width))
        # cv2.imwrite(directory+"/"+images[0]+"/" + str(images[1]).split('.')[0]+".3.jpg",rotated_image_3)
        # cv2.imwrite(directory+"/"+images[0]+"/" + str(images[1]).split('.')[0]+".1.jpg",rotated_image_1)
        # cv2.imwrite(directory+"/"+images[0]+"/" + str(images[1]).split('.')[0]+".2.jpg",rotated_image_2)
        # cv2.imwrite(directory+"/"+images[0]+"/" + str(images[1]).split('.')[0]+".4.jpg",rotated_image_4)
