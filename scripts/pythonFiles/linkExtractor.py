import csv

with open("../photos.csv", "rb") as photosFile:
      photosReader = csv.reader(photosFile, delimiter="\t", lineterminator="\n")
      photosReader.next() #Skipping header line

      for links in photosReader:
          print links[3]
