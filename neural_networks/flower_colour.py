import csv, numpy as np, os
from PIL import Image

from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout, Activation
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.optimizers import SGD

from keras.callbacks import  History, ModelCheckpoint, EarlyStopping
from sklearn.metrics import confusion_matrix, classification_report, auc, roc_curve
from keras import backend as K
from sklearn.cross_validation import StratifiedKFold
import matplotlib.pyplot as plt

def load_labels_and_file_name(tsv_file, directory):
    tsv_list = []
    with open(tsv_file, "rb") as tsv:
        reader = csv.reader(tsv, delimiter="\t", lineterminator="\n")
        for files in reader:
            if os.path.isfile(directory+"/"+files[1]) == True:
                tsv_list.append(files)
            else:
                print "Not found!"
        return tsv_list

def vectorise_image(directory, file_name, height, width):
    loaded_image = Image.open(directory+"/"+file_name)
    loaded_image = loaded_image.resize((height, width), Image.ANTIALIAS)
    vectored_image = np.asarray(loaded_image, dtype="float32")
    return vectored_image

def load_images_using_tsv(directory, tsv_file, height, width):
    labels_file_name = load_labels_and_file_name(tsv_file, directory)
    number_of_images = len(labels_file_name)
    data = np.empty((number_of_images, 3, height, width), dtype="float32")
    data.flatten()
    labels = np.empty((number_of_images, ), dtype=np.dtype("a16"))
    for i, details in enumerate(labels_file_name):
        vectored_image = vectorise_image(directory, details[1], height, width)
        data[i,:,:,:] = [vectored_image[:,:,0]/255,vectored_image[:,:,1]/255,vectored_image[:,:,2]/255]
        labels[i] = details[0]
    return data, labels

def create_model(input_size, number_of_classes):
    model = Sequential()

    model.add(Convolution2D(16, 4, 4, border_mode="valid", input_shape=(3, input_size, input_size)))
    # For visualisation purposes
    convout1 = Activation('relu')
    model.add(convout1)

    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation("relu"))

    model.add(Dense(number_of_classes))
    model.add(Activation("softmax"))

    o = SGD()
    model.compile(loss="categorical_crossentropy", optimizer=o)
    return model, convout1


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

def visualise_first_layer(model, convout1, test_data):
    get_layer_output = K.function([model.layers[0].input], [convout1.get_output(train=False)])
    layer_output = get_layer_output([test_data])[0]

    plt.imshow(layer_output[0,0,:])
    plt.show()

def metrics(all_pred_prob, all_predictions, all_test_labels):
    pass

if __name__ == "__main__":
    pictures_directory = "/home/keo7/Pictures/plant_images" # The directory containing ALL of the plant images.
    labels = "../file_preperation/labels/file.tsv" # The tsv file of which contains the image name and its correlated label. (correlate_photos_to_phenotype.py)

    input_size = (28, 28)

    data, labels= load_images_using_tsv(pictures_directory, labels, input_size[0], input_size[1])
    number_of_classes = len(set(labels))
    print "Data loaded in shape of:", data.shape
    all_pred_prob = []
    all_predictions = []
    all_test_labels = []

    model = None
    stratified_k_fold = StratifiedKFold(labels, n_folds=10, shuffle=True)
    for i,(train, test) in enumerate(stratified_k_fold):
        model, convout1 = create_model(input_size[0], number_of_classes)
        model = train_model(model, data[train], labels[train], number_of_classes)
        pred_prob, predictions = evaluate_model(data[test])
        all_pred_prob.extend(pred_prob)
        all_predictions.extend(predictions)
        all_test_labels.extend(labels[test])
        visualise_first_layer(model, convout1, data[train])

    metrics(all_pred_prob, all_predictions, all_test_labels)
