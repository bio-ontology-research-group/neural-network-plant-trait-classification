import imp

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils

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

    model.add(ZeroPadding2D((1,1), input_shape=(3, input_size[0], input_size[1])))
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
    model.compile(loss="categorical_crossentropy", optimizer=sgd)
    return model

def train_model_and_test(number_of_epochs, number_of_classes, train_data, train_label, augmented_data_generator):
    train_label = np_utils.to_categorical(train_label, number_of_classes)

    for e in range(number_of_epochs):
        progress_bar = generic_utils.Progbar(train_data.shape[0])
        for train_d_batch, train_l_batch in augmented_data_generator(train_data, train_label):
            score = model.train_on_batch(train_d_batch, train_l_batch)
            progress_bar.add(train_d_batch[0], values=[('train loss', score[0])])



if __name__ == "__main__":
    pictures_directory = "/home/keo7/Pictures/plant_images" # The directory containing ALL of the plant images.
    labels = "../file_preperation/labels/small_file.tsv" # The tsv file of which contains the image name and its correlated label. (correlate_photos_to_phenotype.py)

    input_size = (28,28)
    split = 0.8 # 90% of the dataset used for training, leaving 10% out for testing
    number_of_epochs = 20

    (train_data, train_label), (test_data, test_label) = pyvec_api.load_images_with_tsv(pictures_directory, labels, input_size[0], input_size[1], split)
    number_of_classes = len(set(train_label))


    augmented_data_generator = augment_data(train_data)

    model = create_model(input_size, number_of_classes)

    train_model_and_test(number_of_epochs, number_of_classes, train_data, train_label, augmented_data_generator)