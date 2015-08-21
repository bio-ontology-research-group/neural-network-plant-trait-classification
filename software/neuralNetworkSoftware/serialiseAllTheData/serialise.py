import imp
from os.path import expanduser
import os.path as path

dataset_getter = imp.load_source('dataset', '../../pyvec/pyvec/images/dataset.py')
output = imp.load_source('output', '../../pyvec/pyvec/output/pickles.py')
data_maker = imp.load_source('data_maker', '../../pyvec/pyvec/images/new_data.py')

home_directory = expanduser("~")
directory = home_directory + "/datasets/colours/preProcessed"
num_classes = 9
split = 0.7
custom_height = 64
custom_width = 64
save_path = path.abspath(path.join(home_directory))
print save_path

print "Creating more data...\n"

dataset_getter.more_data(directory, custom_height, custom_width)

print "Loading the data...\n"

# I have already preprocessed the data, if you haven't done this please look at the examples
# provided with pyvec: https://github.com/KeironO/Pyvec/blob/master/examples/loadingImages/loading_images.py
train_data, train_label, val_data, val_label, test_data, test_label = dataset_getter.vectorise \
    (directory,num_classes,custom_height,custom_width, split, True)


print('Train data shape:', train_data.shape)
print('Validation data shape:', val_data.shape)
print('Test data shape:', test_data.shape)


print "Saving to file!"

output.pickle_data(save_path, train_data, train_label, val_data, val_label, test_data, test_label)

print "Done!"