# May god have mercy upon my soul.

import csv, re

downloadLinks = []
folderStructure = []

with open("./output.csv", "rb") as traitsf:
    traitsfr = csv.reader(traitsf, delimiter="\t", lineterminator="\n")
    traitsfr.next()

    for traits in traitsfr:
        if traits[6] == "0":
            downloadLinks.append(traits[3])
            folderStructure.append(str(traits[2]) + "\t" + str(traits[5]) + "\n")

out = open("./floweringLinkFile.txt", "wb")
for links in downloadLinks:
    out.write(links + "\n")
out.close()

out = open("./fileStructure.csv", "wb")
for items in folderStructure:
    removedCommas = re.sub(",", "", str(items))
    out.write(removedCommas)

out.close()