import gen_utils
import pattern_utils2
import nom_utils
#from joblib import Parallel, delayed
#import multiprocessing
import numpy as np
import string
import operator
from sklearn.linear_model import LogisticRegression
from copy import deepcopy
import crf_utils
import random

random.seed(4)

class EventExtractor:
	def __init__(self, dependency_type, reduce_deps, token_featurizer, classifier, parallelize_compute, wordlist, event_dict, check_only_nouns=True, check_all_noms=True, num_cores=1, crf_file=False, training=False):
		print "Creating extractor"
		#print dependency_type, reduce_deps, classifier, parallelize_compute, wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, crf_file, training
		self.dependency_type = dependency_type
		self.reduce_deps = reduce_deps
		self.tf = token_featurizer
		self.num_cores = num_cores
		self.parallelize_compute = parallelize_compute
		self.classifier = classifier
		self.wordlist = wordlist
		self.event_dict = event_dict
		self.check_only_nouns = check_only_nouns
		self.check_all_noms = check_all_noms
		self.training = training
		self.sentence_hasher = SentenceHasher(self.tf.w2v_model, 3000)
		self.crf_file = crf_file
		self.crf_tagger = False
		#import pycrfsuite
		#if crf_file:
		#	self.crf_tagger = pycrfsuite.Tagger()
		#	self.crf_tagger.open(crf_file)	
		#else:
		#	self.crf_tagger = False
	def process_token(self, token, sentence, trigger_labels, event_word_counts, hashed_sentence, entity_tagged_sentence):
		#token_features = {}
		if self.training:
			true_event_labels = gen_utils.get_triggers_for_token(token, trigger_labels)
		else:
			true_event_labels = []
		token_back_bigram = self.tf.get_token_back_bigram(token, sentence)
		token_forward_bigram = self.tf.get_token_forward_bigram(token, sentence)
		#token_back_bigram = ""
		#token_forward_bigram = ""
		if self.check_all_noms:		
			#arg_patterns = pattern_utils2.check_token_dep_patterns_plus_wn_types(token, sentence[dependency_type], sentence['tokens'], arg_deps, reduce_deps)
			arg_patterns = []
			#for trigger in trigger_labels:
			#	arg_patterns += gen_utils.get_arg_paths_plus_wn_types(trigger, token, sentence, reduce_deps, dependency_type)[0]
			#if re_gather_deps:
			#	#noun_patterns = gen_utils.get_non_arg_noun_paths(token, sentence, trigger_labels, reduce_deps, dependency_type)[0]		
			noun_patterns = []
			person_patterns = gen_utils.get_trigger_and_nontrigger_person_paths(token, sentence, trigger_labels, self.dependency_type, self.reduce_deps)
			#else:
			#	noun_patterns = []
			#	person_patterns = []
			features = self.tf.construct_features(token, sentence, self.event_dict.keys(), event_word_counts, arg_patterns, noun_patterns, person_patterns, token_back_bigram, token_forward_bigram, hashed_sentence, entity_tagged_sentence)
		else:
			if token['word'].lower() in reverse_dict or token['lemma'] in reverse_dict:
				if token['word'].lower() in reverse_dict:
					check_word = token['word'].lower()
				else:
					check_word = token['lemma']
				if not use_patterns or pattern_utils.bad_dep_patterns_arg_match(token, sentence[self.dependency_type], sentence['tokens'], self.reduce_deps):
					found_match = True
					events = reverse_dict[check_word]
					arg_patterns = pattern_utils2.check_token_dep_patterns_plus_wn_types(token, sentence[self.dependency_type], sentence['tokens'], arg_deps, self.reduce_deps)
					noun_patterns = gen_utils.get_non_arg_noun_paths(token, sentence, trigger_labels, reduce_deps, self.dependency_type)
					features = self.tf.construct_features(token, sentence, events, event_word_counts, arg_patterns, noun_patterns)
		return features, true_event_labels
	def save_sentence(self, filename, s, word, trigger_match, trigger_event, event, this_tp, this_fp, this_tn, this_fn, word_list):
			with open(writefile, 'a') as f:
				writer = csv.writer(f)
				writer.write([filename, s, word, trigger_match, trigger_event, event, this_tp, this_fp, this_tn, this_fn, word_list])
	def process_sentence(self, sentence, trigger_labels, filename, event_word_counts):
		if len(sentence['tokens']) > 150:
			return []
		global missed_triggers
		self.initialize_crf()
		s = gen_utils.create_string_from_tokens(sentence['tokens'])
		s = filter(lambda x: x in string.printable, s)
		sentence_features = []
		sentence_labels = []
		sentence_examples = []
		#hashed_sentence = list(self.sentence_hasher.hash_annotated_sentence(sentence))
		hashed_sentence = []
		#print 'getting ready to crf process sentence'
		crf_features, _ = crf_utils.process_sentence(sentence, [])
		#print 'finished crf_utils process_sentence, getting ready to tag'
		entity_tagged_sentence = self.crf_tagger.tag(crf_features)		
		#print 'finished tagging'
		for token in sentence['tokens']:
			if self.check_only_nouns and token['pos'] not in nom_utils.get_noun_types():
				continue
			# Skip words < three letters (like 'a.' or '1.')
			if len(token['word']) <= 1:
				continue		
			if '<' in token['word'] or '>' in token['word']:
					continue			
			if not self.check_all_noms and token['word'] not in reverse_dict:
				triggers = gen_utils.get_triggers_for_token(token, trigger_labels)
				if len(triggers) == 0:
					continue
				else:
					missed_triggers += len(triggers)
					print token['word']
					continue
			token_features, token_event_labels = self.process_token(token, sentence, trigger_labels, event_word_counts, hashed_sentence, entity_tagged_sentence)
			example = {}
			example['token_features'] = token_features
			example['token_triggers'] = token_event_labels
			example['sentence'] = s
			example['word'] = token['word']
			example['characterOffsetBegin'] = token['characterOffsetBegin']
			example['characterOffsetEnd'] = token['characterOffsetEnd']
			example['filename'] = filename
			example['annotated_sentence'] = sentence
			example['token'] = token
			example['event_word_counts'] = event_word_counts
			sentence_features.append(token_features)
			sentence_labels.append(token_event_labels)
			sentence_examples.append(example)
		#return sentence_features, sentence_labels, s
		return sentence_examples	
	def process_doc(self, doc, trigger_labels, filename):
		if not isinstance(doc, dict):
			return []
		event_word_counts, word_lists = gen_utils.get_event_word_counts(doc, self.wordlist, filename)
		event_word_counts = gen_utils.sort_dict_by_key_return_val(event_word_counts)
		#all_art_events = gen_utils.get_all_events_from_triggers(trigger_labels)
		#all_ace_events = all_ace_events.union(all_art_events)
		doc_features = []
		doc_labels = []
		doc_examples = []
		doc_texts = []
		for sentence in doc['sentences']:
			#sentence_features, sentence_labels, sentence_text = process_sentence(sentence, trigger_labels, filename, event_word_counts)
			#doc_features += sentence_features
			#doc_labels += sentence_labels
			#doc_texts += sentence_text
			doc_examples += self.process_sentence(sentence, trigger_labels, filename, event_word_counts)
		#return doc_features, doc_labels, doc_texts
		if 'use_only_positive' in doc:
			examples_to_use = []
			found_positive = False
			#print "*******"
			#print trigger_labels
			for example in doc_examples:
				if len(example['token_triggers']) > 0:
					examples_to_use.append(example)
					found_positive = True
					#print str(len(example['token_triggers']))
			# if found_positive:
			# 	doc_examples = examples_to_use
			doc_examples = examples_to_use
		return doc_examples	
	def process_docs(self, docs, trigger_labels, filenames):
		num_arts = 0.
		all_examples = []
		#if self.parallelize_compute:
		#	print "Creating parallel extractors"
		#	#extractors = self.get_extractor_copies(len(docs))
		#	print "Created multiple extractors"
		#	examples = Parallel(n_jobs = self.num_cores, backend="threading")(delayed(parallel_process_doc)(extractor, doc, triggers, doc_filename) for doc, triggers, doc_filename in zip(extractors, docs, trigger_labels, filenames))
		#	for example in examples:
		#		all_examples += example
		#else:	
		num_docs = len(docs)
		for i in xrange(num_docs):
			#if num_arts % 5 == 0:
			#	print num_arts / num_docs
			num_arts += 1
			doc = docs[i]
			triggers = trigger_labels[i]
			doc_filename = filenames[i]		
			#doc_features, doc_labels, doc_texts = process_doc(doc, triggers, doc_filename)
			doc_examples = self.process_doc(doc, triggers, doc_filename)
			#all_features += doc_features
			#all_labels += doc_labels
			#all_texts += doc_texts
			# if len(doc_examples) != 1:
			# 	print 'len', str(len(doc_examples))
			# 	print 'num sentences', str(len(doc['sentences']))
			# 	print 'num tokens', str(len(doc['sentences'][0]['tokens']))
			# 	print map(lambda token: token['word'], doc['sentences'][0]['tokens'])
			# 	print 'trigger:'
			# 	print triggers
			# 	print 'found triggers:'
			# 	print map(lambda example: example['token_triggers'], doc_examples)
			all_examples += doc_examples
		#return all_features, all_labels, all_texts
		return all_examples	
	def extract_from_docs_and_save(self, docs, filenames, save_dir):
		print "Preparing to extract from: ", filenames
		trigger_labels = map(lambda x: [], xrange(len(docs)))
		all_examples = []
		#if self.parallelize_compute:
		#	Parallel(n_jobs = self.num_cores)(delayed(parallel_extract_from_doc_and_save)(self, doc, doc_filename, save_dir) for doc, doc_filename in zip(docs, filenames))	
		#else:
		for i, doc in enumerate(docs):
			filename = filenames[i]
			print 'processing doc: ', filename
			self.extract_from_doc_and_save(doc, filename, save_dir)
	def extract_from_doc_and_save(self, doc, filename, save_dir):
		examples = self.process_doc(doc, [], filename)
		if examples == []:
			print "document ", filename, "is broken"
			return
		print filename, len(examples)
		features, labels, sentences, words, labeled_triggers = get_corpus_info(examples)
		pred = self.classifier.predict_proba(features)
		confidences = np.max(pred, 1)
		event_indices = np.argmax(pred, 1)
		events = map(lambda x: self.classifier.classes_[x], event_indices)
		results = zip(events, confidences, examples)
		self.write_doc_results(results, save_dir, filename)
	def extract_from_doc_and_save_with_args(self, doc, filename, save_dir):
		examples = self.process_doc(doc, [], filename)
		if examples == []:
			print "document ", filename, "is broken"
			return
		print filename, len(examples)
		features, labels, sentences, words, labeled_triggers = get_corpus_info(examples)
		pred = self.classifier.predict_proba(features)
		confidences = np.max(pred, 1)
		event_indices = np.argmax(pred, 1)
		events = map(lambda x: self.classifier.classes_[x], event_indices)
		results = zip(events, confidences, examples)
		self.write_doc_results(results, save_dir, filename)		
	def write_doc_results(self, results, save_dir, filename):
		results_to_write = []
		results_to_write = map(lambda x: self.get_output_line(x), results)
		write_filename = save_dir + 'arguments/' + filename
		with open(write_filename, 'wb') as f:
			for item in results_to_write:
				if item:
					f.write(item)
		#for result in results:
		#	results_to_write.append(self.get_output_line(result))
		### output
	def get_output_line(self, result):
		event, confidence, example = result
		if event == "NONE":
			return False
		predicate_justification_offset = str(example['characterOffsetBegin']) + '-' + str(example['characterOffsetEnd'] - 1)
		filename = example['filename']
		role = 'Person'
		additional_arg_justification = "NIL"
		realis = "ACTUAL"
		cas = "John Doe"
		cas_offset = "0-7"
		base_filler_offset = cas_offset
		output_string = filename + '\t' + event + '\t' + role + '\t' + cas + '\t' + cas_offset + '\t' + predicate_justification_offset + '\t' + base_filler_offset + '\t' + additional_arg_justification + '\t' + realis + '\t' + str(confidence)
		line_id = str(self.generate_line_id(output_string))
		output_string = line_id + '\t' + output_string
		return output_string
	def generate_line_id(self, output_string):
		return hash(output_string)
	def get_extractor_copies(self, num_docs):
		initial_copy = deepcopy(self)
		extractors = [initial_copy]
		for i in xrange(self.num_cores - 1):
			extractor_copy = deepcopy(initial_copy)
			extractors.append(extractor)
		times_to_copy = num_docs / num_cores
		extra = num_docs % num_cores
		for i in xrange(times_to_copy):
			extractors += extractors[:num_cores]
		for i in xrange(extra):
			extractors.append(extractors[i])
		return extractors
	def initialize_crf(self):
		if self.crf_tagger == False:
			import pycrfsuite
			self.crf_tagger = pycrfsuite.Tagger()
			self.crf_tagger.open(self.crf_file)






def compile_features_labels(features, labels):
	if len(features) != len(labels):
		print "Problem: Different numbers of features and labels"
	num_tokens = len(features)
	all_features = []
	all_labels = []
	for i in xrange(num_tokens):
		token_features = features[i]
		token_labels = labels[i]
		#true_events = map(lambda x: x['ere_type'] + "." + x['subtype'].lower(), token_labels)
		true_events = map(lambda x: x['event_type'], token_labels)
		if len(true_events) == 0:
			true_events = ["NONE"]
		for label in true_events:
			all_features.append(token_features)
			all_labels.append(label)
	return all_features, all_labels



def compile_features_labels_realis(features, labels):
	if len(features) != len(labels):
		print "Problem: Different numbers of features and labels"
	num_tokens = len(features)
	all_features = []
	all_labels = []
	for i in xrange(num_tokens):
		token_features = features[i]
		token_labels = labels[i]
		#true_events = map(lambda x: x['ere_type'] + "." + x['subtype'].lower(), token_labels)
		true_events = map(lambda x: x['event_type'], token_labels)
		if len(true_events) == 0:
			continue
		realis_labels = map(lambda x: x['realis'], token_labels)
		for label in realis_labels:
			all_features.append(token_features)
			all_labels.append(label)
	return all_features, all_labels


def get_corpus_info(examples):
	all_features = []
	all_labels = []
	all_sentences = []
	all_words = []
	all_labeled_triggers = []
	for example in examples:
		features = example['token_features']
		trigger_info = example['token_triggers']
		sentence = example['sentence']
		word = example['word']
		compiled_features, compiled_labels = compile_features_labels([features], [trigger_info])
		num_labels = len(compiled_labels)
		for i in xrange(num_labels):
			f = compiled_features[i]
			label = compiled_labels[i]
			all_features.append(f)
			all_labels.append(label)
			all_sentences.append(sentence)
			all_words.append(word)
	return all_features, all_labels, all_sentences, all_words, all_labeled_triggers	

def get_corpus_examples_adjusted_to_label_length(examples):
	adjusted_examples = []
	for example in examples:
		features = example['token_features']
		trigger_info = example['token_triggers']
		compiled_features, compiled_labels = compile_features_labels([features], [trigger_info])
		num_labels = len(compiled_labels)
		for i in xrange(num_labels):
			adjusted_examples.append(example)
	return adjusted_examples

def get_corpus_info_realis(examples):
	all_features = []
	all_labels = []
	all_sentences = []
	all_words = []
	all_labeled_triggers = []
	for example in examples:
		features = example['token_features']
		trigger_info = example['token_triggers']
		sentence = example['sentence']
		word = example['word']
		compiled_features, compiled_labels = compile_features_labels_realis([features], [trigger_info])
		num_labels = len(compiled_labels)
		for i in xrange(num_labels):
			f = compiled_features[i]
			label = compiled_labels[i]
			all_features.append(f)
			all_labels.append(label)
			all_sentences.append(sentence)
			all_words.append(word)
	return all_features, all_labels, all_sentences, all_words, all_labeled_triggers	




class RealisExtractor(EventExtractor):
	def __init__(self, dependency_type, reduce_deps, token_featurizer, classifier, parallelize_compute, wordlist, event_dict, check_only_nouns=True, check_all_noms=True, num_cores=1, crf_tagger=False, training=False):
		EventExtractor.__init__(self, dependency_type, reduce_deps, token_featurizer, classifier, parallelize_compute, wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, crf_tagger, training)
		self.verb_types = nom_utils.get_verb_types()
		self.woulda_list = ['would', 'could', 'if', 'may', 'will', 'not', 'should', "won't", 'whether', 'if', 'every', 'often', 'always', 'never', 'none', 'no']
		# list of verb tenses present in sentence
	def process_token(self, token, sentence, trigger_labels, event_word_counts, hashed_sentence):
		true_event_labels = gen_utils.get_triggers_for_token(token, trigger_labels)
		if self.training and len(true_event_labels) == 0:
			return [], []
		#hashed_sentence = list(self.sentence_hasher.hash_annotated_sentence(sentence))
		hashed_sentence = []
		sentence_verbs = map(lambda x: x['pos'], sentence['tokens'])
		sentence_lemmas = map(lambda x: x['lemma'].lower(), sentence['tokens'])
		verb_types_features_list = self.tf.get_dep_pattern_features_list(self.verb_types, sentence_verbs)
		woulda_features_list = self.tf.get_dep_pattern_features_list(self.woulda_list, sentence_lemmas)
		sentence_has_neg = 'neg' in map(lambda x: x['dep'], sentence[self.dependency_type])
		if sentence_has_neg:
			neg_feature = [1]
		else:
			neg_feature = [-1]
		sentence_ners = map(lambda x: 1 if x['ner'] in self.tf.ner_types else 0 , sentence['tokens'])		
		ner_count = [sum(sentence_ners)]
		woulda_count = [sum(woulda_features_list)]
		#return EventExtractor.process_token(self, token, sentence, trigger_labels, event_word_counts, hashed_sentence)
		features = hashed_sentence + verb_types_features_list + woulda_features_list + neg_feature + ner_count + woulda_count
		return features, true_event_labels

def get_NA_and_label_counts(dat):
	docs = dat['training_docs']
	all_labels = dat['training_labels']
	NA_count = 0
	label_count = 0
	for i, doc in enumerate(docs):
		doc_labels = all_labels[i]
		for sentence in doc['sentences']:
			for token in sentence['tokens']:
				triggers = gen_utils.get_triggers_for_token(token, doc_labels)
				if len(triggers) == 0:
					NA_count += 1
				else:
					label_count += len(triggers)
	return NA_count, label_count

def get_ACE_ratios(examples, ace_event_percentages, ace_NA_counts, ace_label_counts, event_list, add_negs=False):
	all_features = []
	all_labels = []
	all_sentences = []
	all_words = []
	all_labeled_triggers = []
	label_counts = {}
	label_type_dict = {}
	for example in examples:
		features = example['token_features']
		trigger_info = example['token_triggers']
		sentence = example['sentence']
		word = example['word']
		compiled_features, compiled_labels = compile_features_labels([features], [trigger_info])
		num_labels = len(compiled_labels)
		for i in xrange(num_labels):
			f = compiled_features[i]
			label = compiled_labels[i]
			if label not in label_type_dict:
				label_type_dict[label] = {}
				label_type_dict[label]['all_features'] = []
				label_type_dict[label]['all_labels'] = []
				label_type_dict[label]['all_sentences'] = []
				label_type_dict[label]['all_words'] = []
				label_type_dict[label]['all_labeled_triggers'] = []
			label_type_dict[label]['all_features'].append(f)
			label_type_dict[label]['all_labels'].append(label)
			label_type_dict[label]['all_sentences'].append(sentence)
			label_type_dict[label]['all_words'].append(word)
			if label not in label_counts:
				label_counts[label] = 0.
			label_counts[label] += 1
	ace_total_examples = ace_label_counts + ace_NA_counts
	ace_label_percentage = float(ace_label_counts) / ace_total_examples
	if "NONE" not in ace_event_percentages:
		for event, percentage in ace_event_percentages.items():
			ace_event_percentages[event] = percentage * ace_label_percentage
		ace_event_percentages['NONE'] = float(ace_NA_counts) / ace_total_examples
	if not add_negs:
		ace_event_percentages['NONE'] = 0
		# normalize back to 1:
		sum_percentages = 0.
		for label, percent in ace_event_percentages.items():
			sum_percentages += percent
		for label, percent in ace_event_percentages.items():
			ace_event_percentages[label] = percent / sum_percentages
	print "ace percentages:"
	print ace_event_percentages
	num_to_add = float('inf') # number of search examples we can use
	for event, count in label_counts.items():
		if event not in ace_event_percentages:
			continue
		can_add = float(count) / ace_event_percentages[event]
		if can_add < num_to_add:
			num_to_add = int(can_add)	
	search_event_add_counts = {}
	search_event_percentages = {}
	for event, count in label_counts.items():
		if event not in ace_event_percentages:
			continue
		num_to_add_this_event = num_to_add * ace_event_percentages[event]
		print "adding", event, num_to_add_this_event
		search_event_add_counts[event] = num_to_add_this_event
		percent_of_this_event = float(num_to_add_this_event) / count
		search_event_percentages[event] = percent_of_this_event
		indices_to_add = random.sample(range(int(count)), int(num_to_add_this_event))
		all_features += [label_type_dict[event]['all_features'][i] for i in indices_to_add]
		print 'all features length:', len(all_features)
		all_labels += [label_type_dict[event]['all_labels'][i] for i in indices_to_add]
		all_sentences += [label_type_dict[event]['all_sentences'][i] for i in indices_to_add]
		all_words += [label_type_dict[event]['all_words'][i] for i in indices_to_add]
	print "percentages of search data added:"
	print search_event_percentages	
	return all_features, all_labels, all_sentences, all_words, all_labeled_triggers	


def limit_counts(examples, ace_label_counts, max_percent=1.):
	event_examples_dict = {}
	examples_to_return = []
	for example in examples:
		first_trigger_info = example['token_triggers'][0] # should only be one
		event = first_trigger_info['event_type']
		if event not in event_examples_dict:
			event_examples_dict[event] = []
		event_examples_dict[event].append(example)
	for event, event_examples in event_examples_dict.items():
		event_count = ace_label_counts[event]
		max_to_add = int(event_count * max_percent)
		if max_to_add < len(event_examples):
			examples_to_use = random.sample(event_examples, max_to_add)
		else:
			examples_to_use = event_examples
		examples_to_return += examples_to_use
	return examples_to_return


