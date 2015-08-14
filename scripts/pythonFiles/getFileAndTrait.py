import csv,re
from collections import defaultdict

def create_traits_dictonary(traits_file, trait):
    traits_dictonary = {}
    with open(traits_file,"rb") as traits_f:
        traits_reader = csv.reader(traits_f, delimiter="\t", lineterminator="\n")
        traits_reader.next() #Skipping header line
        previous_trait = None #Removal of weird double traits.
        for rows in traits_reader:
            if previous_trait != None and rows[3] != previous_trait:
                if rows[3] == trait:
                    if rows[1] not in traits_dictonary:
                        traits_dictonary[rows[1]] = []
                    traits_dictonary[rows[1]].append(rows)
            previous_trait = rows[3]
    return traits_dictonary


def create_photos_list(photos_file, traits_dictonary):
    photos_list = []
    with open(photos_file, "rb") as photosFile:
        photosReader = csv.reader(photosFile, delimiter="\t", lineterminator="\n")
        photosReader.next()

        for photos in photosReader:
            id = photos[1]
            try:
                if photos[5] != "0" and photos[6] !="1": #not a fruit and is flowering
                    removedCommas = re.sub(",","", str(photos[2])) #Files have commas in them.
                    photos_list.append(str(traits_dictonary[id]) + str(removedCommas))
            except:
                pass
        return photos_list

def remove_crap(photos_list):
    uncrapified_array = []
    for photos in photos_list:
        removed_crap = re.sub("]]",", ", photos)
        removed_crap = re.sub("\[\[" , ",", removed_crap)
        removed_crap = re.sub(", " , "\t", removed_crap)
        removed_crap = re.sub("," , "", removed_crap)
        removed_crap = re.sub("\'" , "", removed_crap)
        uncrapified_array.append(removed_crap)
    return uncrapified_array

def make_csv_array(uncrapified_array):
    new_csv = csv.reader(uncrapified_array, delimiter="\t", lineterminator="\n")
    return new_csv

def save_file_structure(new_csv, trait):
    out = open("./createdFiles/"+trait+".csv", "wb")
    for items in new_csv:
        if items[4] != trait:
            out.write(items[4]+","+items[5]+"\n")
    out.close()


if __name__ == "__main__":
    trait = "Flower - Stamen number"
    traits_file = "./traitsAndPhotos/traits.csv"
    photos_file = "./traitsAndPhotos/photos.csv"
    traits_dictonary = create_traits_dictonary(traits_file, trait)
    photos_list = create_photos_list(photos_file, traits_dictonary)
    uncrapified_array = remove_crap(photos_list)
    new_csv = make_csv_array(uncrapified_array)
    save_file_structure(new_csv, trait)

