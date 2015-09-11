import nltk, csv, re

array = []

with open("/home/osheak/labeledsentences.tsv", "rb") as file:
    file_reader = csv.reader(file, delimiter="\t", lineterminator="\n")
    for rows in file_reader:
        if rows[0] == "1":
            # Remove all non-ASCII characters.
            rows[3] = re.sub(r'[^\x00-\x7F]+',' ', rows[3])
            rows[3] = re.sub(r'</?[QqEe]>?>', ' ', rows[3])
            # Lowercase everything because I'm lazy.
            array.append([x.lower() for x in rows])
            #break to only read a single line in, for debugging purposes.

for idx, rows in enumerate(array):
    rows.append(idx)
    rows[3] = nltk.word_tokenize(rows[3])
    rows[3] = nltk.pos_tag(rows[3])


for rows in array:
    for postagged in rows[3]:
        if postagged[0] == rows[1]:
            print postagged[0] + "\t" + postagged[1] + "\tQ"
        elif postagged[0] == rows[2]:
            print postagged[0] + "\t" + postagged[1] + "\tE"
        else:
            print postagged[0] + "\t" + postagged[1] + "\t0"
    print "\n"