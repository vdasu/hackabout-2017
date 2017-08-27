Instructions:

1. Run svm.py in python3
2. svm.py returns output.txt and prints the accuracy score.
3. output.txt contains line numbers and the corresponding class name.

If the tester wishes to load the trained classifier, the sem_eval_classifier.joblib.pkl in the data directory has to be read using sklearn's joblib function.
(The above instructions are to be followed if testers utilise the features extracted  from the training data provided. If a different dataset is used, follow the description)

Description:
1. sentence.py initialises a Sentence class which stores extracted features
2. lexical_features.py and nominal_features.py extract features and store the features in a .pkl file.
3. Run svm.py after feature extraction 
4. Accuracy score is printed and the output is written to output.txt.

NOTE: The pickled training data features are stored in cleaned.pkl and the test data features which are extracted after training are stored in cleaned_test_full.pkl. If the tester wishes to train and test the classifier on new data, he/she is required to change the address of the file being read in line 68 of lexical_features.py. The address of the extracted features which are pickled can be modified by changing the arguments in line 140 of lexical_features.py. The lexical_features.py script has to be run twice i.e once for storing the extracted training data features and then to store the extracted test data features.

Dependencies:
1. nltk
2. sklearn
3. spacy
4. numpy

Methodology:
	The features being extracted from the given dataset along with the rationale behind using said feature is given below- 
		- Distance between nominals - This is because certain classes have lower distance between the two nominals whereas other classes have a general tendency to have a larger distance between nominals.
		- Average of the nominal vectors and the vectors of words between nominals(using SpaCy model to vectorize) -  This is because SVM is much more optimized and effective at dealing with numerical data and this also gives the model a sense of the general meaning of the sentence.
		- Term Frequencies of the POS Tags of all the words - This gives the model an indication of grammatical structure of the sentence along with the semantic relationship between the nominals.

		

Results:
	Using the given methodology and training the SVM Model on the given training data we were able to achieve an accuracy of 69.04% on the testing dataset.We found to have achieved the best accuracy using a LINEAR kernel for SVM with the Regularization paramter (C) set to 6000 
	

Further Improvements:
	- Term Frequency dictionary could have been more comprehensive.
	- Word2Vec Model could have been trained on a larger corpus 
	- The Lowest Common Hypernym and the Jaccard Distance would have been very valuable and insightful features to have had however we were not able to integrate these into our model due to time constraints.
	- Ensemble methods could have been used to provide for a more robust model and improve further accuracy.



