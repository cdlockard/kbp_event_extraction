import nom_utils
import gen_utils
import readACE4
import pickle
import numpy as np
import gensim
import os
import scipy
from nltk.corpus import wordnet as wn

dep_mapping = gen_utils.get_dep_mapping()


synset_dict = {}
synset_dict['meet'] = [1,2,3,8, 13]
synset_dict['attack']= [0, 9, 14] # 12?
synset_dict['die'] = [3]
synset_dict['buy'] = [1] # maybe more
synset_dict['elect'] = [1]

#synset_dict['meet'] = [0]
#synset_dict['attack']= [0] # 12?
#synset_dict['die'] = [0]
#synset_dict['buy'] = [0] # maybe more
#synset_dict['elect'] = [0]

synset_dict['bankrupt'] = [0]
synset_dict['close'] = [5]
synset_dict['merge'] = [2]
synset_dict['found'] = [1,2]
synset_dict['demonstrate'] = [3]
synset_dict['broadcast'] = [0]
synset_dict['correspond'] = [2]
synset_dict['acquit'] = [0]
synset_dict['appeal'] = [2,4]
synset_dict['arrest'] = [0,2]
synset_dict['indict'] = [0]
synset_dict['convict'] = [2]
synset_dict['execute'] = [0]
synset_dict['extradite'] = [0]
synset_dict['fine'] = [0,1]
synset_dict['pardon'] = [0,1,2,4]
synset_dict['parole'] = [2,3]
synset_dict['sentence'] = [1,2,3]
synset_dict['sue'] = [1]
synset_dict['hearing'] = [0,8]
synset_dict['birth'] = [0,1,2,4,5]
synset_dict['divorce'] = [0,2]
synset_dict['injure'] = [0,2]
synset_dict['marry'] = [0,1]
synset_dict['manufacture'] = [0,1,2,5]
#synset_dict['move'] = [6]
synset_dict['move'] = [5]
synset_dict['quit'] = [1]
synset_dict['nominate'] = [0,1]
synset_dict['hire'] = [1,2]



def get_token(index, tokens):
	return tokens[index - 1]

def get_dep(dep, reduce_deps):
	dep = dep.split(':')[0]
	if reduce_deps:
		return dep_mapping[dep]
	return dep

def follow_path(current_token_index, dep, direction, dependencies, reduce_deps = False):
	for dependency in dependencies:
		if get_dep(dependency['dep'], reduce_deps) == dep:
			if direction == 'up' and dependency['dependent'] == current_token_index:
				return dependency['governor']
			if direction == 'down' and dependency['governor'] == current_token_index:
				return dependency['dependent']				
	return -1

def get_pattern_end_token_index(token, dependencies, dep_pattern, reduce_deps = False):
	if dep_pattern == False:
		return -1
	dependencies = dependencies
	current_token_index = token['index']
	#print dep_pattern
	for direction, dep in dep_pattern:
		current_token_index = follow_path(current_token_index, dep, direction, dependencies, reduce_deps)
		#print dep, direction, current_token_index
		if current_token_index == -1:
			return -1
	return current_token_index


def dep_pattern_exists(token, dependencies, dep_pattern, reduce_deps = False):
	end_token_index = get_pattern_end_token_index(token, dependencies, dep_pattern, reduce_deps)
	if end_token_index == -1:
		return False
	return True

def get_pattern_wn_types(token, dependencies, tokens, dep_pattern, reduce_deps = False):
	token_lemma = dep_pattern[-1][1]
	dep_pattern = dep_pattern[:len(dep_pattern) - 1]
	wn_type = dep_pattern[-1][1]
	dep_pattern = dep_pattern[:len(dep_pattern) - 1]	
	if 'person' in wn_type:
		wn_type = "PERSON"
	elif 'location' in wn_type:
		wn_type = 'LOCATION'
	elif 'time' in wn_type:
		wn_type = 'TIME'
	elif 'group' in wn_type:
		wn_type = "ORGANIZATION"
	elif 'noun' in wn_type:
		wn_type = 'noun'
	elif 'verb' in wn_type:
		wn_type = 'verb'
	if token['ner'] != 0:
		wn_type = token['ner']
	end_token_index = get_pattern_end_token_index(token, dependencies, dep_pattern, reduce_deps)
	if end_token_index == -1:
		return False
	if end_token_index == 0 and wn_type == "ROOT":
		return True
	matching_token = get_token(end_token_index, tokens)
	#wn_types = gen_utils.get_hypernym_list(matching_token['lemma'])
	token_wn_type = gen_utils.get_wn_type(matching_token['lemma'])
	#print matching_token['word'], token_wn_type, wn_type
	if token_wn_type == wn_type:
		return True
	return False


def check_token_dep_patterns_plus_wn_types(token, dependencies, tokens, dep_patterns, reduce_deps = False):
	matching_patterns = []
	for dep_pattern in dep_patterns:
		if dep_pattern == False:
			continue
		pattern_list = create_pattern_list_from_pattern_string(dep_pattern)
		match = get_pattern_wn_types(token, dependencies, tokens, pattern_list, reduce_deps)
		if match:
			matching_patterns.append(dep_pattern)
	return matching_patterns




def check_token_dep_patterns(token, dependencies, dep_patterns, reduce_deps = False):
	matching_patterns = []
	for dep_pattern in dep_patterns:
		if dep_pattern == False:
			continue
		pattern_list = create_pattern_list_from_pattern_string(dep_pattern)
		if dep_pattern_exists(token, dependencies, pattern_list, reduce_deps):
			matching_patterns.append(dep_pattern)
	return matching_patterns


#dep_pattern_exists(token, dependencies, pattern, True)
#get_pattern_end_token_index(token, dependencies, new_pattern, True)



def create_pattern_list_from_pattern_string(pattern_string):
	pattern_list = []
	pattern_comps = pattern_string.split(' ')
	if len(pattern_comps) == 1:
		return []
	# use[1:] because 'token' is first thing
	for i, pattern_comp in enumerate(pattern_comps[1:]):
		if i % 2 == 0:
			# direction
			new_pattern_comp = [0, 0]
			new_pattern_comp[0] = pattern_comp			
		else:
			new_pattern_comp[1] = pattern_comp
			pattern_list.append(new_pattern_comp)
	return pattern_list

bad_dep_patterns = []
#bad_dep_patterns.append('token up compound')
#bad_dep_patterns.append('token up mod')
bad_dep_patterns.append('token down mod')
#bad_dep_patterns.append('token up mod up mod')
#bad_dep_patterns.append('token up nsubj up compound')
#bad_dep_patterns.append('token up mod up compound')

def get_bad_dep_patterns():
	return bad_dep_patterns

def bad_dep_patterns_match(token, dependencies, reduce_deps = False):
	for pattern_string in bad_dep_patterns:
		pattern_list = create_pattern_list_from_pattern_string(pattern_string)
		if dep_pattern_exists(token, dependencies, pattern_list, reduce_deps):
			return True
	return False

def bad_dep_patterns_arg_match(token, dependencies, tokens, reduce_deps = False):
	for pattern_string in bad_dep_patterns:
		pattern_list = create_pattern_list_from_pattern_string(pattern_string)
		index = get_pattern_end_token_index(token, dependencies, pattern_list, reduce_deps)
		if index != -1 and index != 0:
			token_match = tokens[index - 1]
			if token_match['pos'] in nom_utils.get_noun_types():# or token_match['pos'] in ["PRP", "PRP$", "SYM", "CD"]:
				return True
	return False	
	







#########################################
#########################################
#########################################
#########################################

## Wordnet





#########################################
#########################################
#########################################
#########################################

## Construct Features

## token
## preceding & following tokens
## governor and dependent tokens
## governor and dependent dep types
## NER types present in sentences
## dep paths to NER things
## "likely trigger dep paths" & WN classes of those end things
## verb token
## verb POS
## helper verb dep relations/lemmas/POS
## word list features
## doc-level features for potential event
## dependency features = how?
## realis classification?
##

#filename = 'nominals/forJames/default-with_verbs.pkl'
#from pycorenlp import StanfordCoreNLP
test_sentence = "There will be a meeting on Thursday in Beijing. This is another sentence about things. And also other things."
#nlp = StanfordCoreNLP('http://localhost:9000')
#doc = test_sentence
#annotated_text = nlp.annotate(doc, properties={'timeout': 9999, 'annotators': 'tokenize, ssplit, pos, depparse, lemma, ner', 'outputFormat': 'json'})	
#sentence = annotated_text['sentences'][0]
#token = sentence['tokens'][0]

test_vocab = test_sentence.split(' ')
test_vocab_dict = {}
test_vocab_dict['lemmas'] = test_vocab
test_vocab_dict['words'] = test_vocab
# "model" dict with things



class TokenFeaturizer:
	def __init__(self, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, dep_pattern_features, noun_dep_pattern_features, person_dep_pattern_features, bigram_feature_set, w2v_model, testing=False):
		print "Creating TokenFeaturizer"
		#print dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, dep_pattern_features, noun_dep_pattern_features, person_dep_pattern_features, bigram_feature_set, testing
		if testing:
			self.w2v_model = w2v_model
		else:
			#self.w2v_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(os.path.join(os.path.dirname("__file__"), 'GoogleNews-vectors-negative300.bin'), binary=True)
			self.w2v_model = w2v_model
		self.dependency_type = dependency_type
		self.reduce_deps = reduce_deps
		self.use_lemmas = use_lemmas
		self.use_w2v = use_w2v
		#if testing:
		training_vocab = test_vocab_dict
		#else:
		#	training_vocab = self.get_training_vocab()
		self.reverse_dict = self.get_noms(noms_filename)
		self.event_dict = nom_utils.get_event_dict(self.reverse_dict)
		self.trigger_vocab = self.get_trigger_vocab()
		self.dep_pattern_features = dep_pattern_features
		self.noun_dep_pattern_features = noun_dep_pattern_features
		self.person_dep_pattern_features = person_dep_pattern_features
		self.bigram_feature_set = bigram_feature_set
		if use_lemmas:
			self.all_word_vocab = list(training_vocab['lemmas'])
		else:
			self.all_word_vocab = list(training_vocab['words'])
		self.all_word_vocab = list(set(self.all_word_vocab + [""]))
		#self.all_word_vocab = training_vocab['words']
		#self.all_lemma_vocab = training_vocab['lemmas']
		#### Vocabs
		self.dep_vocab = self.get_dep_vocab()
		self.w2v_vocab = []
		self.arg_dep_patterns = []
		self.bad_dep_patterns = []
		self.ner_types = ['ORGANIZATION', 'LOCATION', 'PERSON', 'MISC', 'MONEY', 'PERCENT', 'DATE', 'TIME', 'NUMBER', 'ORDINAL', 'DURATION', 'SET']
		self.pos_vocab = gen_utils.get_all_pos_tags()
		self.testing = testing
		self.bigram_hash_count = 10 # previously 10000
		self.entity_types = gen_utils.entity_type_list
		#self.bigram_hash_count = 10000
	def get_noms(self, filename):
		import pickle
		with open(filename, 'rb') as f:
			reverse_dict = pickle.load(f)	
		return reverse_dict
	def get_trigger_vocab(self):
		return self.reverse_dict.keys()
	## Do I want lemma or full word?
	def get_token_word(self, token):
		if self.use_lemmas:
			return token['lemma']
		else:
			return token['word']
	def get_governor_words_and_dep_relations(self, token, sentence):
		token_index = token['index']
		governor_words = []
		governor_dep_types = []
		for dependency in sentence[self.dependency_type]:
			if dependency['dependent'] == token_index:
				gov_token = self.get_token_by_index(dependency['governor'], sentence['tokens'])
				gov_word = self.get_token_word(gov_token)
				governor_words.append(gov_word)
				gov_dep_type = dependency['dep'].split(':')[0]
				if self.reduce_deps:
					gov_dep_type = dep_mapping[gov_dep_type]
				governor_dep_types.append(gov_dep_type)
		return governor_words, governor_dep_types
	def get_dependent_words_and_dep_relations(self, token, sentence):
		# variable names are reversed, copied from get_governor_words_and_dep_relations
		token_index = token['index']
		governor_words = []
		governor_dep_types = []
		for dependency in sentence[self.dependency_type]:
			if dependency['governor'] == token_index:
				gov_token = self.get_token_by_index(dependency['dependent'], sentence['tokens'])
				gov_word = self.get_token_word(gov_token)
				governor_words.append(gov_word)
				gov_dep_type = dependency['dep'].split(':')[0]
				if self.reduce_deps:
					gov_dep_type = dep_mapping[gov_dep_type]
				governor_dep_types.append(gov_dep_type)
		return governor_words, governor_dep_types
	def create_ner_dict(self):
		ner_dict = {}
		ner_dict['ORGANIZATION'] = 0
		ner_dict['LOCATION'] = 0
		ner_dict['PERSON'] = 0
		ner_dict['MISC'] = 0
		ner_dict['MONEY'] = 0
		ner_dict['PERCENT'] = 0
		ner_dict['DATE'] = 0
		ner_dict['TIME'] = 0
		ner_dict['NUMBER'] = 0
		ner_dict['ORDINAL'] = 0
		ner_dict['DURATION'] = 0
		ner_dict['SET'] = 0
		return ner_dict
	def get_ner_counts(self, sentence):
		ner_dict = self.create_ner_dict()
		for token in sentence['tokens']:
			ner = token['ner']
			if ner != 'O':
				ner_dict[ner] += 1
		ner_features = []
		for ner in self.ner_types:
			ner_features.append(ner_dict[ner])
		return ner_features
	def get_potential_args_and_features(self, token, sentence):
		matching_patterns = dep_patterns_match(self.arg_dep_patterns)
		return matching_patterns
	def count_bad_dep_patterns(self, token, sentence):
		matching_patterns = dep_patterns_match(self.bad_dep_patterns)
		return len(matching_patterns)
	def construct_features(self, token, sentence, events, doc_event_word_counts, matching_dep_patterns, matching_noun_dep_patterns, matching_person_dep_patterns, back_bigram, forward_bigram, hashed_sentence, entity_tagged_sentence):
		# get previous token
		#doc_event_word_counts = []
		features = []
		if self.use_w2v:
			token_word = self.get_w2v(token['word'])
		else:
			token_word = self.featurize_bow(token['word'], self.trigger_vocab)
		if token['index'] == 1:
			if self.use_w2v:
				prev_token_word = self.get_w2v("")
			else:
				prev_token_word = self.featurize_bow("", self.all_word_vocab)
			prev_token_pos = self.featurize_bow("NONE", self.pos_vocab)
		else:
			prev_token = self.get_token_by_index(token['index'] - 1, sentence['tokens'])
			if self.use_w2v:
				prev_token_word = self.get_w2v(prev_token['word'])
			else:
				prev_token_word = self.featurize_bow(prev_token['word'], self.all_word_vocab)
			prev_token_pos = self.featurize_bow(prev_token['pos'], self.pos_vocab)
		# get next token
		if token['index'] == len(sentence['tokens']):
			if self.use_w2v:
				next_token_word = self.get_w2v('')
			else:
				next_token_word = self.featurize_bow("", self.all_word_vocab)
			next_token_pos = self.featurize_bow("NONE", self.pos_vocab)
		else:
			next_token = self.get_token_by_index(token['index'] + 1, sentence['tokens'])
			if self.use_w2v:
				next_token_word = self.get_w2v(next_token['word'])
			else:
				next_token_word = self.featurize_bow(next_token['word'], self.all_word_vocab)
			next_token_pos = self.featurize_bow(next_token['pos'], self.pos_vocab)
		# get governor token and dep type
		governor_words, governor_dep_types = self.get_governor_words_and_dep_relations(token, sentence)
		if self.use_w2v:
			governor_word_features = self.get_w2v_from_list(governor_words)
		else:
			governor_word_features = self.featurize_bow_list(governor_words, self.all_word_vocab)
		governor_dep_features = self.featurize_bow_list(governor_dep_types, self.dep_vocab)
		dependent_words, dependent_dep_types = self.get_dependent_words_and_dep_relations(token, sentence)
		if self.use_w2v:
			dependent_word_features = self.get_w2v_from_list(dependent_words)
		else:
			dependent_word_features = self.featurize_bow_list(dependent_words, self.all_word_vocab)
		dependent_dep_features = self.featurize_bow_list(dependent_dep_types, self.dep_vocab)
		#ner_counts = self.get_ner_counts(sentence)
		ner_counts = []
		all_nom_features = []
		w2v_event_averages = []
		for potential_event in self.event_dict.keys():
			#print potential_event
			if potential_event in events:		
				if token['word'] in self.reverse_dict:
					nom_features = self.featurize_noms(token['word'], potential_event)
				elif token['lemma'] in self.reverse_dict:
					nom_features = self.featurize_noms(token['lemma'], potential_event)
				else:
					nom_features = map(lambda x: 0, self.featurize_noms('meeting', potential_event))
			else:
				nom_features = map(lambda x: 0, self.featurize_noms('meeting', potential_event))					
			w2v_event_avg = self.get_w2v_distance_from_event_list(token['word'], potential_event)
			all_nom_features += nom_features
			w2v_event_averages.append(w2v_event_avg)
		#w2v_event_avg = self.get_w2v_distance_from_list(token['word'], synset_dict.keys())	
		#w2v_event_averages = []
		#all_nom_features = []
		if self.testing:
			print len(all_nom_features)
		#dep_pattern_features_list = self.get_dep_pattern_features_list(self.dep_pattern_features, matching_dep_patterns)
		noun_dep_pattern_features_list = self.get_dep_pattern_features_list(self.noun_dep_pattern_features, matching_noun_dep_patterns)
		dep_pattern_features_list = []
		#noun_dep_pattern_features_list = []
		person_dep_pattern_features_list = self.get_dep_pattern_features_list(self.person_dep_pattern_features, matching_person_dep_patterns)
		#back_bigram_features_list = self.get_dep_pattern_features_list(self.bigram_feature_set, [back_bigram])
		#forward_bigram_features_list = self.get_dep_pattern_features_list(self.bigram_feature_set, [forward_bigram])
		back_bigram_features_list = self.hash_into_list(back_bigram, self.bigram_hash_count)
		forward_bigram_features_list = self.hash_into_list(forward_bigram, self.bigram_hash_count)
		#back_bigram_features_list = []
		#forward_bigram_features_list = []
		ner_type = self.get_dep_pattern_features_list(self.ner_types, [token['ner']])
		sentence_entity_types = self.get_entity_types(entity_tagged_sentence)
		entity_types_features_list = self.get_dep_pattern_features_list(self.entity_types, sentence_entity_types)
		### Everything below is for testing
		# ner_counts = []
		# ner_type = []
		# sentence_entity_types = []
		# entity_types_features_list = []
		# w2v_event_averages = []
		# all_nom_features = []		
		# back_bigram_features_list = []
		# forward_bigram_features_list = []
		# person_dep_pattern_features_list = []
		# noun_dep_pattern_features_list = []
		# dep_pattern_features_list = []
		# dependent_word_features = []
		# dependent_dep_features = []
		# governor_word_features = []
		# governor_dep_features = []
		doc_event_word_counts = []
		return token_word + prev_token_word + prev_token_pos + next_token_word + next_token_pos + ner_counts + dependent_word_features + dependent_dep_features + governor_word_features + governor_dep_features + all_nom_features + doc_event_word_counts + w2v_event_averages + dep_pattern_features_list + noun_dep_pattern_features_list + person_dep_pattern_features_list + back_bigram_features_list + forward_bigram_features_list + hashed_sentence + ner_type + entity_types_features_list
	def featurize_bow(self, word, vocab):
		bin_list = map(lambda x: 0, vocab)
		for i, term in enumerate(vocab):
			if term == word:
				bin_list[i] = 1
		return bin_list
	def featurize_bow_list(self, word_list, vocab):
		bin_list = map(lambda x: 0, vocab)
		for i, term in enumerate(vocab):
			if term in word_list:
				bin_list[i] += 1
		return bin_list	
	def get_training_vocab(self):
		training_vocab = {}
		training_vocab['words'] = set()
		training_vocab['lemmas'] = set()
		dat = readACE3.get_ACE_data(False, False)
		training_docs = dat['training_docs']
		training_labels = dat['training_labels']
		training_filenames = dat['training_filenames']
		for doc in training_docs:
			for sentence in doc['sentences']:
				for token in sentence['tokens']:
					training_vocab['words'].add(token['word'])
					training_vocab['lemmas'].add(token['lemma'])
		return training_vocab
	def get_dep_vocab(self):
		dep_mapping = gen_utils.get_dep_mapping()
		if self.reduce_deps:
			dep_vocab = set(dep_mapping.values())
		else:
			dep_vocab = set(dep_mapping.keys())
		return list(dep_vocab)
	def featurize_noms(self, word, event):
		nom_features = []
		if word in self.reverse_dict and event in self.reverse_dict[word]:
			nom = self.reverse_dict[word][event]
			for key, val in nom.items():				
				if key in ['event', 'word', 'pos']:
					continue
				else:
					nom_features.append(val)
		else:
			# steal format from real nom
			nom = self.reverse_dict['meeting']['Contact.Meet']
			for key, val in nom.items():		
				if key in ['event', 'word', 'pos']:
					continue
				else:
					nom_features.append(0)
		return nom_features
	def get_token_by_index(self, index, tokens):
		return tokens[index - 1]
	def get_w2v(self, word):
		if word in self.w2v_model:
			return list(self.w2v_model[word])
		else:
			return map(lambda x: 0, xrange(300))
	def get_w2v_from_list(self, wordlist):
		if len(wordlist) == 0:
			return self.get_w2v('')
		w2v = np.zeros(300)
		for word in wordlist:
			this_w2v = self.get_w2v(word)
			np.add(w2v, this_w2v)
		return list(np.multiply(w2v, 1./len(wordlist)))
	def get_w2v_distance_from_event_list(self, word, event):
		if word not in self.w2v_model:
			#return list(np.ones(300))
			return 1
		#sum_distance = np.zeros(300)
		sum_distance = 0.
		event_count = 0
		for event_word in self.event_dict[event]:
			if event_word in self.w2v_model:
				#sum_distance = np.add(sum_distance, self.w2v_model['event_word'])
				#sum_distance += scipy.spatial.distance.cosine(self.w2v_model[word], self.w2v_model[event_word])
				sum_distance += gen_utils.cosine_distance(self.w2v_model[word], self.w2v_model[event_word])
				event_count += 1
		if event_count == 0:
			return 1
		else:
			#return np.multiply(sum_distance, 1./event_count)
			return sum_distance / event_count
	# def get_w2v_distance_from_list(self, word, word_list):
	# 	if word not in self.w2v_model:
	# 		#return list(np.ones(300))
	# 		return 1
	# 	#sum_distance = np.zeros(300)
	# 	sum_distance = 0.
	# 	event_count = 0
	# 	for event_word in word_list:
	# 		if event_word in self.w2v_model:
	# 			#sum_distance = np.add(sum_distance, self.w2v_model['event_word'])
	# 			#sum_distance += scipy.spatial.distance.cosine(self.w2v_model[word], self.w2v_model[event_word])
	# 			sum_distance += gen_utils.cosine_distance(self.w2v_model[word], self.w2v_model[event_word])
	# 			event_count += 1
	# 	if event_count == 0:
	# 		return 1
	# 	else:
	# 		#return np.multiply(sum_distance, 1./event_count)
	# 		return sum_distance / event_count			
	def new_pattern_dict(self, dep_patterns):
		pattern_dict = {}
		for pattern in dep_patterns:
			pattern_dict[pattern] = 0
		return pattern_dict
	def get_dep_pattern_features_list(self, all_dep_patterns, matching_dep_patterns):
		pattern_dict = self.new_pattern_dict(all_dep_patterns)
		for pattern in matching_dep_patterns:
			if pattern in pattern_dict:
				pattern_dict[pattern] = 1
		#return gen_utils.sort_dict_by_key_return_val(pattern_dict)
		return pattern_dict.values()
	def get_token_back_bigram(self, token, sentence):
		if token['index'] == 1:
			return "ROOT " + self.get_token_word(token)
		return self.get_token_word(token) + " " + self.get_token_word(self.get_token_by_index(token['index'] - 1, sentence['tokens']))
	def get_token_forward_bigram(self, token, sentence):
		if token['index'] == len(sentence['tokens']):
			return self.get_token_word(token) + " END"
		return self.get_token_word(token) + " " + self.get_token_word(self.get_token_by_index(token['index'] + 1, sentence['tokens']))
	def hash_into_list(self, val, size):
		hash_list = [0] * size
		hash_list[hash(val) % size] = 1
		return hash_list
	def get_entity_types(self, entity_tagged_sentence):
		types = []
		for tag in entity_tagged_sentence:
			if tag == 'NONE':
				continue
			types.append(tag[2:])
		return types


#class ArgFeaturizer:
#	def __init__(self, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, dep_pattern_features, noun_dep_pattern_features, w2v_model, testing=False):


#tf = TokenFeaturizer(dependency_type, True, False, True, filename, True)
#tf.construct_features(token, sentence, "Contact.Meet")