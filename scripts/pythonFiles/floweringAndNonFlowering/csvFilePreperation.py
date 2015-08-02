import csv, re

traitsDictonary = {}
photosArray = []

with open("../traits.csv","rb") as traitsFile:
    traitsReader = csv.reader(traitsFile, delimiter="\t", lineterminator="\n")
    traitsReader.next() #Skipping header line
    traitsDictonary = {rows[1] : rows for rows in traitsReader}

with open("../photos.csv", "rb") as photosFile:
    photosReader = csv.reader(photosFile, delimiter="\t", lineterminator="\n")
    photosReader.next()

    for photos in photosReader:
        id = photos[1]
        try:
            if photos[5] != "0":
                photosArray.append(str(traitsDictonary[id]) + str(photos))
        except:
            pass


out = open("../output.csv", "wb")
for stuff in photosArray:
    # Quick and Dirty
    removeBracket1 = re.sub("\[", "", str(stuff))
    removeBracket2 = re.sub("\]", "", str(removeBracket1))

    out.write(removeBracket2 + "\n")
out.close()