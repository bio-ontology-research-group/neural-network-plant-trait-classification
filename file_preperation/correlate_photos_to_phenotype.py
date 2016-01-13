'''

'''


import csv,re


def load_tsv(file_name):
    with open(file_name, "rb") as tsv_file:
        reader = csv.reader(tsv_file, delimiter="\t", lineterminator="\n")
        reader.next()
        loaded_list = list(reader)
        return loaded_list

def return_only_trait(trait, traits_file):
    trait_info = []
    previous_id = None
    for i, rows in enumerate(traits_file):
        if rows[3] == trait and previous_id != rows[1]:
            previous_id = rows[1]
            trait_info.append([rows[1], rows[-1]])
    return trait_info

def traits_to_dict(trait_info):
    traits_dict = dict(trait_info)
    return traits_dict

def correlate_photos_file_to_traits_info(photos_file, traits_dict):
    trait_and_file_list = []
    for rows in photos_file:
        species_id = rows[1]
        if rows[5] != "0" and rows[6] != "1": # If not a fruit and is flowering
            try:
                trait_and_file_list.append([str(traits_dict[species_id]), rows[2]])
            except:
                pass
    return trait_and_file_list

def list_to_csv_file(trait, trait_file_and_list):
    save_file = open("./labels/"+trait+".tsv", "wb")
    csv_writer = csv.writer(save_file, delimiter="\t", lineterminator="\n")

    for files in trait_file_and_list:
        csv_writer.writerow(files)

def get_download_links(photos_file):
    save_file = open("./download_links.txt", "wb")
    for photos in photos_file:
        save_file.writelines("%s\n" % photos[3])

if __name__ == "__main__":
    trait = "Flower - Colour"

    # Loading in the traits file
    traits_file = load_tsv("./files_to_parse/traits.csv")
    # Returning a list of traits
    trait_info = return_only_trait(trait, traits_file)
    # Convert list to dictonary
    traits_dict = traits_to_dict(trait_info)

    # Loading in the photos file
    photos_file = load_tsv(("./files_to_parse/photos.csv"))
    # If you don't have the download links then uncomment the following line.
    # get_download_links(photos_file)
    # Correlating the photo ID to the trait ID
    trait_and_file_list = correlate_photos_file_to_traits_info(photos_file, traits_dict)

    # Save to afile named the trait.tsv
    list_to_csv_file(trait, trait_and_file_list)
