import imp, numpy as np

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD
from keras.utils import np_utils, generic_utils

from keras.callbacks import  History, ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve
from keras import backend as K
from sklearn.cross_validation import StratifiedKFold


pyvec_api = imp.load_source('api', './pyvec/pyvec/core/api.py')

def create_model(input_size, number_of_classes):
    model = Sequential()

    model.add(Convolution2D(32, 3, 3, border_mode="same", input_shape=(3, input_size, input_size)))
    model.add(Activation("relu"))


    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))

    model.add(Dense(number_of_classes))
    model.add(Activation("softmax"))

    o = SGD()
    model.compile(loss="categorical_crossentropy", optimizer=o)
    return model


def train_model(model, train_data, train_label, number_of_classes):
    train_label = to_categorical(train_label, number_of_classes)
    early_stopper = EarlyStopping(monitor="val_loss", patience=100)
    model.fit(train_data, train_label, batch_size=32, nb_epoch=1000, verbose=1, validation_split=0.1, callbacks=[early_stopper])

    return model

def evaluate_model(test_data):
    pred_prob = model.predict_proba(test_data, verbose=0).tolist()
    predictions = model.predict_classes(test_data, verbose=0).tolist()

    return pred_prob, predictions

def to_categorical(y, nb_classes=None):
    '''Convert class vector (integers from 0 to nb_classes)
    to binary class matrix, for use with categorical_crossentropy.
    '''
    y = np.asarray(y, dtype='a16')
    for i, labels in enumerate(y):
        y[i] = "".join(str(ord(char)) for char in y[i])
    y = np.asarray(y, dtype='float32')
    if not nb_classes:
        nb_classes = np.max(y)+1
    Y = np.zeros((len(y), nb_classes))
    for i in range(len(y)):
        Y[i] = 1.
    return Y

if __name__ == "__main__":
    pictures_directory = "/home/keo7/Pictures/small_plant_images" # The directory containing ALL of the plant images.
    labels = "../file_preperation/labels/small_file.tsv" # The tsv file of which contains the image name and its correlated label. (correlate_photos_to_phenotype.py)

    input_size = (28, 28)

    (data, labels), (__,__) = pyvec_api.load_images_with_tsv(pictures_directory, labels, input_size[0], input_size[1], 1)
    number_of_classes = len(set(labels))

    model = None
    stratified_k_fold = StratifiedKFold(labels, n_folds=10, shuffle=True)
    for i,(train, test) in enumerate(stratified_k_fold):
        model = create_model(input_size[0], number_of_classes)
        model = train_model(model, data[train], labels[train], number_of_classes)
        pred_prob, predictions = evaluate_model(data[test])