import csv

with open("/home/osheak/herb/english-eq.txt") as file:
    file_reader = csv.reader(file, delimiter="\t", lineterminator="\n")

    for rows in file_reader:
        print rows[2], "\t", rows[1], "\t", rows[3], "\t",rows[6]
