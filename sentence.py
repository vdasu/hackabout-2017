class Sentence:

	"""
	Constructor for the sentence datatype

	Args:
	    nominals: Tuple (e1, e2)
	    sentence: Array of strings representing the entire sentence
	    nominal_distance: Number of words between the nominals 
        pos_nominals: POS tags of nominals
        pos_words: POS tags of all words in the sentence
        stem_words: Stems of words between nominals
        label: Target label for the sentence
		indices Indexes of e1 and e2 in the sentence - Tuple(index_e1, index_e2)
	Returns:
	    Returns a Sentence obj
	"""
	def __init__(self, nominals, sentence, nominal_distance, pos_nominals, pos_words, stem_words, label, pos_between_nominals,vector_avg,vector_avg_words):
		self.e1=nominals[0]
		self.e2=nominals[1]
		self.sentence = sentence
		self.nominal_distance = nominal_distance
		self.pos_nominals = pos_nominals
		self.pos_words = pos_words
		self.stem_words = stem_words
		self.label = label
		self.pos_between_nominals = pos_between_nominals
		self.vector_avg = vector_avg
		self.vector_avg_words = vector_avg_words
		# self.index_e1 = indices[0]
		# self.index_e2 = indices[1]

	# def __init__(self, nominals, sentence, nominal_distance, pos_nominals, pos_words, stem_words, pos_between_nominals):
	# 	self.e1=nominals[0]
	# 	self.e2=nominals[1]
	# 	self.sentence = sentence
	# 	self.nominal_distance = nominal_distance
	# 	self.pos_nominals = pos_nominals
	# 	self.pos_words = pos_words
	# 	self.stem_words = stem_words
	# 	self.pos_between_nominals = pos_between_nominals

	
	def get_nominals(self):
		return (self.e1,self.e2)

	def create_feature_dict(self):
		feature_dict = {'e1':self.e1, 'e2':self.e2}
		feature_dict.update({'words:'+sentence_word:True for sentence_word in self.sentence})
		feature_dict.update({'nom_dist':self.nominal_distance})
		feature_dict.update({'nom_pos:'+pos_nominal:True for pos_nominal in self.pos_nominals})
		feature_dict.update({'words_pos:'+pos_word:True for pos_word in self.pos_words})
		feature_dict.update({'words_stem:'+stem_word:True for stem_word in self.stem_words})
		return feature_dict

	def sentence_dict(self):
		word_dict = {'words:'+word:True for word in self.sentence} 		
		return word_dict
