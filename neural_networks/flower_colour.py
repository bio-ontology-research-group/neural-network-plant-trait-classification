import imp

pyvec_api = imp.load_source('api', './pyvec/pyvec/core/api.py')

def create_model():
    pass



if __name__ == "__main__":
    pictures_directory = "/home/keo7/Pictures/plant_images" # The directory containing ALL of the plant images.
    labels = "../file_preperation/labels/file.tsv" # The tsv file of which contains the image name and its correlated label. (correlate_photos_to_phenotype.py)

    input_size = (28, 28)
    split = 0.9 # 90% of the dataset used for training, leaving 10% out for testing

    (train_data, train_label), (test_data, test_label) = pyvec_api.load_images_with_tsv(pictures_directory, labels, input_size[0], input_size[1], split)