import csv,re
from collections import defaultdict

def create_traits_dictonary(traits_file):
    traits_dictonary = {}
    with open(traits_file,"rb") as traits_f:
        traits_reader = csv.reader(traits_f, delimiter="\t", lineterminator="\n")
        traits_reader.next() #Skipping header line
        previous_trait = None #Removal of weird double traits.
        for rows in traits_reader:
            if previous_trait != None and rows[3] != previous_trait:
                if rows[3] == "Flower - Colour":
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

def make_photos_list_useful(photos_list):
    final_array = []
    for photos in photos_list:
        removed_crap = re.sub("]]",", ", photos)
        removed_crap = re.sub("\[\[" , ",", removed_crap)
        final_array.append(removed_crap)
        print removed_crap
    return final_array


if __name__ == "__main__":
    traits_file = "../traits.csv"
    photos_file = "../photos.csv"
    traits_dictonary = create_traits_dictonary(traits_file)
    photos_list = create_photos_list(photos_file, traits_dictonary)
    final_array = make_photos_list_useful(photos_list)
