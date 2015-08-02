# Fairly simple file that only returns flowering photographs.

import csv

with open("./output.csv", "rb") as traitsf:
    traitsfr = csv.reader(traitsf, delimiter="\t", lineterminator="\n")
    traitsfr.next()

    for traits in traitsfr:
        if traits[6] == "0":
            print traits[2] + "\t" + traits[5]
