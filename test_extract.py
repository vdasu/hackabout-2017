from nltk.tag import pos_tag
from nltk.stem.snowball import SnowballStemmer
from nltk.tokenize import word_tokenize
import re
from sentence import *  # For the Sentence object
import _pickle as pickle
import numpy as np

def readData():
    nominals = []
    words_between_nominals = []
    word_prefixes = []
    pos_nominals = []  # Stores POS of nominals
    pos_words = []  # Stores POS of words between nominals
    stem_words = []  # Stores stems of words between nominals
    pos_sent = []  # Stores POS of all words in sentence
    sentences = []  # Holds all the sentences from the text files, but the sentences are not cleaned
    sent2 = []  # Sent2 holds all the cleaned sentences in double quotes
    Sent = []  # An array of Sentence objects, from sentency.py
    indices = []
    # Stores string of starting POS tags of words between nominals
    pos_between_nominals = []

    # Reads the file off training data
    file = open("data/TEST_FILE.txt", "r")
    data = file.readlines()

    # Loop for collecting sentences and storing in sentences[]
    for line in data:
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
                             pos_between_nominals[i]))
    # Pickles file.
    pickle.dump(Sent, open('data/cleaned_test.pkl', 'wb'), protocol=2)


readData()