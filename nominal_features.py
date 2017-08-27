from nltk.corpus import wordnet as wn
from nltk.metrics import jaccard_distance
from nltk.corpus import brown
from nltk.util import ngrams
from collections import Counter
import string

"""
    NOTE: synsets 'pos' argument accepts only four possible attributes. 
    They are as follows:
        VERB
        NOUN
        ADV
        ADJ
    POS tags of the words have to be classified into the above categories
    before the functions can be used.
"""

"""
    Dictionary that stores POS of POS tags
"""
pos = {
    'JJ': 'ADJ',
    'NN': 'NOUN',
    'RB': 'ADV',
    'VB': 'VERB'
}

"""
    hypernym takes input of a list containg (word,pos_tag) tuple pairs
    and return a list of hypernyms.
"""


def hypernym(words):
    hypernyms = []
    for word in words:
        arg = wn.synsets(word[0], pos=getattr(wn, pos[word[1][:2]]))[0]
        hypernyms.append(arg.hypernyms()[0].name().split('.')[0])
    return hypernyms


"""
    lowest_common_hypernym takes input a list containing the 
    [(nominal1,pos_tag),(nominal2,pos_tag)] tuple pairs and 
    returns lowest common hypernym.
"""


def lowest_common_hypernym(words):
    lch = []
    for word in words:
        arg_1 = wn.synsets(words[0][0], pos=getattr(wn, pos[words[0][1]]))[0]
        arg_2 = wn.synsets(words[1][0], pos=getattr(wn, pos[words[1][1]]))[0]
    lch.append(arg_1.lowest_common_hypernyms(arg_2)[0].name().split('.')[0])
    return lch


"""
    jaccard_common returns two values. First value is a list containing the four most
    common objects of each nominal. Second value is the Jaccard distance of the two
    nominal sets.
"""


def jaccard_common(nominals):
    sents = brown.sents()
    sents_no_punct = []
    for sent in sents:
        sents_no_punct.append(
            [''.join(c for c in s if c not in string.punctuation) for s in sent])
    sents_no_punct = [words for sent in sents_no_punct for words in sent]
    sents_no_punct = [word for word in sents_no_punct if word]
    five_grams = ngrams(sents_no_punct, 5)
    e1_words, e2_words = [], []
    for five_gram in five_grams:
        if nominals[0] in five_gram:
            for word in five_gram:
                if word != nominals[0]:
                    e1_words.append(word)
        elif nominals[1] in five_gram:
            for word in five_gram:
                if word != nominals[1]:
                    e2_words.append(word)
    e1_top, e2_top = [], []
    e1_count = Counter(e1_words)
    e2_count = Counter(e2_words)
    e1_top = [word[0] for word in e1_count.most_common(4)]
    e2_top = [word[0] for word in e2_count.most_common(4)]
    return [e1_top, e2_top], jaccard_distance(set(Counter(e1_words).keys()), set(Counter(e2_words).keys()))
