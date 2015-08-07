# It works, so don't judge. It literally took 3 minutes to write.

import csv, os, re

traitsDict = {}
photosDict = {}
traitsArray = []
traitsRow = []
replacements = {'[':'', ']':'', '\'' : '', ',' : ''}

with open("../traits.csv", "rb") as traitsFile:
    traitsReader = csv.reader(traitsFile, delimiter="\t", lineterminator="\n")
    traitsReader.next()
    previousTrait = None
    for traits in traitsReader:
        if previousTrait != None and traits[3] != previousTrait:
            if traits[3] == "Flower - Colour":
                traitsDict.setdefault(traits[1].strip(),[]).append(traits[4].strip(),)
        previousTrait = traits[3]

traitsDict.items()
traitsArray=[(k,v) for k, v in traitsDict.iteritems()]

###################################################################################################
with open("./tempfile.csv", "wb") as f:
	writer = csv.writer(f, delimiter = "\t", lineterminator = "\n")
	writer.writerows(traitsArray)
f.close()
with open("./tempfile.csv") as infile, open("./output.csv", "wb") as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)
outfile.close()
infile.close()
###################################################################################################

with open("../photos.csv", "rb") as photosFile:
    photosReader = csv.reader(photosFile, delimiter="\t", lineterminator="\n")
    for photos in photosReader:
        photosDict.setdefault(photos[1].strip(),[]).append(photos[2].strip(),)

with open("./output.csv", "rb") as newTraitsFile:
    newTraitsReader = csv.reader(newTraitsFile, delimiter="\t", lineterminator="\n")
    newTraitsReader.next()
    for traits in newTraitsReader:
        id = traits[0]
        try:
                traitsRow.append(photosDict[id] + traits)
        except:
            pass


with open("./tempfile.csv", "wb") as coloursStructure:
	writer = csv.writer(coloursStructure, delimiter = "\t", lineterminator = "\n")
	writer.writerows(traitsRow)
coloursStructure.close()

with open("./tempfile.csv") as infile, open("./colours.csv", "wb") as outfile:
    for line in infile:
        for src, target in replacements.iteritems():
            line = line.replace(src, target)
        outfile.write(line)
outfile.close()
infile.close()

os.remove('./tempfile.csv')
os.remove("./output.csv")