import imp

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils

from keras.preprocessing.image import ImageDataGenerator

pyvec_api = imp.load_source('api', './pyvec/pyvec/core/api.py')

def augment_data(train_data):
    augmented_data_generator = ImageDataGenerator(
        rotation_range=20,
        horizontal_flip=True
    )

    augmented_data_generator.fit(train_data)
    return augmented_data_generator

def create_model(input_size, number_of_classes):
    model = Sequential()

    model.add(ZeroPadding2D((1,1), input_shape=(input_size[0], input_size[1])))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3))
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 3, 3))
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 3, 3))
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation("relu"))

    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation("relu"))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 3, 3))
    model.add(Activation("relu"))


    model.add(MaxPooling2D((2,2), strides=(2,2)))

    model.add(Flatten())
    model.add(Dense(4096,))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))

    model.add(Dense(number_of_classes))
    model.add(Activation("softmax"))

    sgd = SGD()
    model.compile(loss="multiple_crossentropy", optimizer=sgd)
    return model

def train_model_and_test():
    pass



if __name__ == "__main__":
    pictures_directory = "/home/keo7/Pictures/plant_images" # The directory containing ALL of the plant images.
    labels = "../file_preperation/labels/file.tsv" # The tsv file of which contains the image name and its correlated label. (correlate_photos_to_phenotype.py)

    input_size = (224, 224)
    split = 0.9 # 90% of the dataset used for training, leaving 10% out for testing

    (train_data, train_label), (test_data, test_label) = pyvec_api.load_images_with_tsv(pictures_directory, labels, input_size[0], input_size[1], split)
    number_of_classes = len(set(train_label))

    augmented_data_generator = augment_data(train_data)

    model = create_model(input_size, number_of_classes)