import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
import _pickle as pickle
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import ShuffleSplit
from nominal_features import *
from sklearn.feature_extraction import DictVectorizer
from sentence import *
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.externals import joblib
FEATURE_LEN = 8000

Sent = pickle.load(open("data/cleaned.pkl", "rb"))
SentTest = pickle.load(open("data/cleaned_test_full.pkl","rb"))
# print(Sent[0].create_feature_dict())

# Target


# Features

# Order of features: (nominal_distance, pos_nominal, stem_word)

vectorizer = CountVectorizer()
corpus = []
pos_corpus = []
number_of_words = []
corpus_test = []
pos_corpus_test = []
number_of_words_test = []
word2vec = []
word2vec_test = []
words_vec = []
words_vec_test = []
for s in Sent:
    corpus.append(s.sentence)
    pos_corpus.append(" ".join(str(x) for x in s.pos_words))
    number_of_words.append(s.nominal_distance)
    word2vec.append(s.vector_avg)
    words_vec.append(s.vector_avg_words)

no_of_words = []
no_of_words.append(number_of_words)

n_o_w = np.transpose(np.asarray(no_of_words))


for s in SentTest:
    corpus_test.append(s.sentence)
    pos_corpus_test.append(" ".join(str(x) for x in s.pos_words))
    number_of_words_test.append(s.nominal_distance)
    word2vec_test.append(s.vector_avg)
    words_vec_test.append(s.vector_avg_words)
no_of_words_test = []
no_of_words_test.append(number_of_words_test)

n_o_w_test = np.transpose(np.asarray(no_of_words_test))

sentences = vectorizer.fit_transform(corpus)
vec_sentences = sentences.toarray()  # Vectors of all the sentences in the corpus
vec_sentences_test = vectorizer.transform(corpus_test).toarray()  # Vectors of all the sentences in the corpus

pos_words = vectorizer.fit_transform(pos_corpus)
vec_pos_words = pos_words.toarray()  # Vectors of all pos tags in corpus
#sentences = vectorizer.fit_transform(corpus)


#pos_words = vectorizer.fit_transform(pos_corpus)
vec_pos_words_test = vectorizer.transform(pos_corpus_test).toarray()  # Vectors of all pos tags in corpus

X_test = None
X = None
Y = np.empty(8000)
Y_test = np.empty(2717)	
i = 0
print(np.shape(vec_sentences))
print(np.shape(vec_pos_words))
print(np.shape(n_o_w))

print(np.shape(vec_sentences_test))
print(np.shape(vec_pos_words_test))
print(np.shape(n_o_w_test))

X = np.append(vec_sentences, vec_pos_words, axis=1)
X = np.append(X, n_o_w, axis=1)
X = np.append(X, word2vec, axis=1)
X = np.append(X, words_vec, axis=1)

X_test = np.append(vec_sentences_test, vec_pos_words_test, axis=1)
X_test = np.append(X_test, n_o_w_test, axis=1)
X_test = np.append(X_test, word2vec_test, axis=1)
X_test = np.append(X_test, words_vec_test, axis=1)
for i, s in enumerate(SentTest):
    Y_test[i] = s.label

for i, s in enumerate(Sent):
    Y[i] = s.label


print(X.shape)
print(Y.shape)
print(X_test.shape)
print(Y_test.shape)

X = X[:FEATURE_LEN]
Y = Y[:FEATURE_LEN]


'''
clf = SVC(kernel='linear', C=1.5, verbose=True)
cv = ShuffleSplit(n_splits=5, test_size=0.5, random_state=0)
scores = cross_val_score(clf, X, Y, cv=cv)
print(scores)
print('IsFinite:', np.isfinite(Y).all())
print('isnull:', np.isnan(Y).any().any())
'''
clf = SVC(C=5000)
clf.fit(X, Y)

Y_pred = clf.predict(X_test)
print(str(accuracy_score(Y_test, Y_pred))+" -> "+str(5000))

#transferring output to file
labels = ['Other','Cause-Effect','Component-Whole','Entity-Destination','Product-Producer','Entity-Origin','Member-Collection','Message-Topic','Instrument-Agency','Content-Container']
file = open('output.txt','w')
i = 8001
Y_pred = Y_pred.astype(np.int64)
print(Y_pred)
for num in Y_pred:
	file.write(str(i)+"\t"+str(labels[num])+"\n")
	i=i+1
file.close()


_ = joblib.dump(clf,"data/sem_eval_classifier.joblib.pkl",compress=9)

'''
hypernyms = []

for s in Sent[:3]:
    hypernyms.append(lowest_common_hypernym([(s.get_nominals()[1],
                                              s.pos_nominals[1]),
                                             (s.get_nominals()[1],
                                              s.pos_nominals[1])]))
print(Sent[1].get_nominals()[1], Sent[1].pos_nominals[1])
print(hypernyms[:10])
'''
