from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import re
from sentence import *  # For the Sentence object
import _pickle as pickle
import numpy as np
import spacy

""" Returns an integer from 0-9 for the various classes
               Other : 0
        Cause-Effect : 1
     Component-Whole : 2
  Entity-Destination : 3
    Product-Producer : 4
       Entity-Origin : 5
   Member-Collection : 6
       Message-Topic : 7
   Content-Container : 8
   Instrument-Agency : 9
"""
nlp = spacy.load('en')

def getClass(label):
    if label == 'Other':
        return 0
    elif label == 'Cause-Effect':
        return 1
    elif label == 'Component-Whole':
        return 2
    elif label == 'Entity-Destination':
        return 3
    elif label == 'Product-Producer':
        return 4
    elif label == 'Entity-Origin':
        return 5
    elif label == 'Member-Collection':
        return 6
    elif label == 'Message-Topic':
        return 7
    elif label == 'Content-Container':
        return 8
    elif label == 'Instrument-Agency':
        return 9


def readData():

    nominals = []
    words_between_nominals = []
    word_prefixes = []
    pos_nominals = []  # Stores POS of nominals
    pos_words = []  # Stores POS of words between nominals
    stem_words = []  # Stores stems of words between nominals
    pos_sent = []  # Stores POS of all words in sentence
    labels = []  # labels will hold all the relations per sentence, such as 'Instrument-Agency'
    class_labels = []  # Holds the labels after conversion to integers, see the getClass() method
    sentences = []  # Holds all the sentences from the text files, but the sentences are not cleaned
    sent2 = []  # Sent2 holds all the cleaned sentences in double quotes
    Sent = []  # An array of Sentence objects, from sentency.py
    vector_avg = [] #Stores average of the word vectors
    vector_avg_words = [] #Stores average of vectors of words between nominals
    indices = []
    # Stores string of starting POS tags of words between nominals
    pos_between_nominals = []

    # Reads the file off training data
    file = open("data/TRAIN_FILE.TXT", "r")
    data = file.readlines()

    # Loop for extracting relations and storing in labels[]
    for i in range(1, len(data), 4):
        if 'Other' in data[i]:
            labels.append('Other')
        else:
            labels.append(re.findall(r'(.*?)\(', data[i])[0])

    # Loop for converting relations to integer labels and storing in class_labels[]
    for label in labels:
        class_labels.append(getClass(label))

    # Loop for collecting sentences and storing in sentences[]
    for line in data:
        if line[0].isdigit():
            sentences.append(line)

    stemmer = SnowballStemmer("english")
    # Extracting required features from each sentence in sentences[]
    for sent in sentences:
        e1 = re.findall(r'<e1>(.*?)<\/e1>', sent)[0]  # Extracts tag <e1>
        e2 = re.findall(r'<e2>(.*?)<\/e2>', sent)[0]  # Extracts tag <e2>
        between_nominals = re.search(r'</e1>(.*?)<e2>', sent).group(1)
        words = between_nominals.split()
        nominal_words = re.search(r'</e1>(.*?)<e2>', sent).group(1)
        nominals.append((e1, e2))
        e1_toks = word_tokenize(e1)
        e2_toks = word_tokenize(e2)
        pos_between_nominals.append(
            ''.join([i[1][0] for i in pos_tag(word_tokenize(between_nominals))]))
        pos_nominals.append((pos_tag(e1_toks)[0][1], pos_tag(e2_toks)[0][1]))
        nominal_words_toks = word_tokenize(nominal_words)
        pos_words.append(([i[1] for i in pos_tag(nominal_words_toks)]))
        stem_words.append([stemmer.stem(i) for i in nominal_words_toks])
        words_between_nominals.append((len(words)))
        a = sent[:sent.find('<')].count(' ')
        b = sent[:sent.rfind('>')].count(' ')
        indices.append((a, b))
        prefixes = []
        doc = nlp(e1+" "+e2)
        vector_avg.append(doc.vector)
        doc_words = nlp(nominal_words)
        vector_avg_words.append(doc_words.vector)

        for i in range(1, len(words) - 1):
            prefixes.append(words[i][:5])

        # Not sure if word prefixes are necessary, kept it anyway
        word_prefixes.append(prefixes)

    # Just to check whether the number of nominals equals sentences
    print(len(sentences), len(nominals))

    # Populates sent2 with the 'cleaned up' sentence
    for sent in sentences:
        sent2.append(
            re.sub(r'<.*?>', '', re.search(r'(\"(.*?)\")', sent).group(2)))

    for sent in sent2:
        pos_sent.append([i[1] for i in pos_tag(word_tokenize(sent))])

    # Populates sent=[] with required Sentence objects
    for i in range(len(nominals)):
        Sent.append(Sentence(nominals[i], sent2[i],
                             words_between_nominals[i],
                             pos_nominals[i], pos_sent[i], stem_words[i],
                             class_labels[i], pos_between_nominals[i],vector_avg[i],vector_avg_words[i]))

    print(class_labels[8], labels[8])
    # Pickles file.
    pickle.dump(Sent, open('data/cleaned.pkl', 'wb'), protocol=2)


readData()
