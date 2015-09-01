import nltk, csv, re

array = []

with open("/home/osheak/sample.txt", "rb") as file:
    file_reader = csv.reader(file, delimiter="\t", lineterminator="\n")
    for rows in file_reader:
        if rows[0] != "0":
            # Remove all non-ASCII characters.
            rows[3] = re.sub(r'[^\x00-\x7F]+',' ', rows[3])
            # Remove all crap from the sentences.
            rows[3] = re.sub('</Q>>', '', rows[3])
            rows[3] = re.sub('</q>>', '', rows[3])
            rows[3] = re.sub('<q>', '', rows[3])
            rows[3] = re.sub('<Q>', '', rows[3])
            rows[3] = re.sub('<E>', '', rows[3])
            rows[3] = re.sub('</E>', '', rows[3])
            # Lowercase everything because I'm lazy.
            array.append([x.lower() for x in rows])

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
