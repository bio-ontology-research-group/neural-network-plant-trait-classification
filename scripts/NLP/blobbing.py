import random, csv, re
from textblob.classifiers import NaiveBayesClassifier, DecisionTreeClassifier, PositiveNaiveBayesClassifier
from textblob import TextBlob
random.seed(1337)

sentences = []
unl_sentences = []

with open("/home/osheak/labeledsentences.tsv", "rb") as file:
    file_reader = csv.reader(file, delimiter="\t", lineterminator="\n")
    for rows in file_reader:
        rows[3] = re.sub(r'[^\x00-\x7F]+','', rows[3]).lstrip()
        # rows[3] = re.sub(r'</?[QqEe]>?>', '', rows[3])
        sentences.append((rows[3], rows[0]))

total_number_images = len(sentences)

split = 0.90 # 0.9 == 90% training, 10% testing.

num_training_data = len(sentences) * split

random.shuffle(sentences)

train_data = sentences[:int(num_training_data)]
test_data = sentences[int(num_training_data):]

print "Training on", len(train_data), "sentences."
print "Testing on", len(test_data), "sentences."

nbc = NaiveBayesClassifier(train_data)
nbcaccuracy = nbc.accuracy(test_data)
print "Accuracy:", nbcaccuracy


with open("/home/osheak/toclassify.txt", "rb") as file:
    file_reader = csv.reader(file, delimiter="\t", lineterminator="\n")
    for rows in file_reader:
        rows[2] = re.sub(r'[^\x00-\x7F]+','', rows[2]).lstrip()
        # rows[3] = re.sub(r'</?[QqEe]>?>', '', rows[3])
        unl_sentences.append((rows[2], rows[2]))

for sents in unl_sentences:
    blob = TextBlob(sents[0], classifier=nbc)
    print blob.classify(), "\t", sents[0]
