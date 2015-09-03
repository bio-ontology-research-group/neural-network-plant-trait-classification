import nltk, csv, re

array = []

tagged_array = []

with open("/home/osheak/labeledsentences.tsv", "rb") as file:
    file_reader = csv.reader(file, delimiter="\t", lineterminator="\n")
    for rows in file_reader:
        if rows[0] != "asdasdadasd":
            # Remove all non-ASCII characters.
            rows[3] = re.sub(r'[^\x00-\x7F]+','', rows[3])
            rows[3] = re.sub(r'</?[QqEe]>?>', '', rows[3])
            # Lowercase everything because I'm lazy.
            array.append([x.lower() for x in rows])
            #break to only read a single line in, for debugging purposes.

for rows in array:
    print rows[3].lstrip()
