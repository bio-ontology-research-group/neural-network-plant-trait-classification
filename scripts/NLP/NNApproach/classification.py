import csv, re, numpy
from keras.preprocessing import sequence
from keras.optimizers import SGD, RMSprop, Adagrad
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dropout, Activation, TimeDistributedDense
from keras.layers.recurrent import LSTM, GRU



def build_model():
    print "Building a Model:"
    model = Sequential()
    model.add(TimeDistributedDense(30, 512, init='he_normal'))
    model.add(Dropout(0.5))
    model.add(LSTM(512, 128, activation='hard_sigmoid', return_sequences=True))
    model.add(LSTM(128, 6, activation='hard_sigmoid'))
    model.add(Activation('hard_sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', class_mode="categorical")
    return model

def load_data():
    print "Loading Data:"
    data = []
    labels = []
    with open("/home/osheak/labeledsentences.tsv", "rb") as file:
        file_reader = csv.reader(file, delimiter="\t", lineterminator="\n")
        for rows in file_reader:
            rows[3] = re.sub(r'[^\x00-\x7F]+',' ', rows[3]).lstrip()
            data.append(rows[3])
            labels.append(rows[0])
    return data, labels


def create_train_and_test(data, labels):
    print "Creating training and testing data"
    train_data = numpy.asarray(data[:50])
    train_data = train_data[:, numpy.newaxis, :]
    train_label = np_utils.to_categorical(labels[:50])

    test_data = numpy.asarray(data[50:])
    test_data = test_data[:, numpy.newaxis, :]
    test_labels = np_utils.to_categorical(labels[50:])

    return train_data, train_label, test_data, test_labels

if __name__ == '__main__':
    model = build_model()
    data, labels = load_data()
    train_data, train_label, test_data, test_labels = create_train_and_test(data, labels)

    print "Training"
    model.fit(train_data, train_label, nb_epoch=10, batch_size=10)
