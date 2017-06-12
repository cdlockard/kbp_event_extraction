

from lsh_mod.LocalitySensitiveHash import LocalitySensitiveHash
import numpy as np




class SentenceHasher:
	def __init__(self, w2v_model, num_projections):
		self.w2v_model = w2v_model
		self.num_projections = num_projections
		self.lsh_model = LocalitySensitiveHash(300, num_projections)
		#self.local_lsh_model = LocalitySensitiveHash(300, 600)
	def hash_annotated_sentence(self, sentence):
		features = np.array(map(lambda x: 0, xrange(0, self.num_projections)), dtype='float64')
		word_count = 0
		for token in sentence['tokens']:
			word = token['word']
			#w2v = np.array(map(lambda x: 0, range(0,300)), dtype='float64')
			if word in self.w2v_model:
				w2v = self.w2v_model[word]
			elif word.lower() in self.w2v_model:
				w2v = self.w2v_model[word.lower()]
			elif len(word) > 2 and (word[0].upper() + word[1:]) in self.w2v_model:
				w2v = self.w2v_model[word[0].upper() + word[1:]]
			else:
				continue			
			word_count += 1
			feature = self.lsh_model.get_hashed_value_bin(w2v)
			features = np.add(features, feature)
		if word_count > 0:
			normalized_reg_features = np.multiply(features, 1. / word_count)
		else:
			normalized_reg_features = features
		return normalized_reg_features
	def hash_local_context(self, token, sentence, window=2):
		token_index = token['index']
		window_min = min(token_index - window, 1) - 1
		window_max = max(token_index + window, len(sentence['tokens']))
		local_context = sentence['tokens'][window_min : window_max] 
		local_sentence_dict = {}
		local_sentence_dict['tokens'] = local_context
		#return avg_word_vec(local_sentence_dict)
		#return self.hash_annotated_sentence(local_sentence_dict)
		return self.avg_word_vec(local_sentence_dict)
	def avg_word_vec(self, sentence):
		features = np.array(map(lambda x: 0, xrange(0, 300)), dtype='float64')
		word_count = 0
		for token in sentence['tokens']:
			word = token['word']
			if word in self.w2v_model:
				w2v = self.w2v_model[word]
				np.add(features, w2v)
				word_count += 1
		if word_count > 0:
			normalized_reg_features = np.multiply(features, 1. / word_count)
		else:
			normalized_reg_features = features
		return normalized_reg_features
