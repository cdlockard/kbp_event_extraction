import pattern_utils2
import extraction_utils
from copy import deepcopy
import random
from joblib import Parallel, delayed
import multiprocessing
import time
import gensim, os
from sklearn.linear_model import LogisticRegression
import math
import arg_utils
import gen_utils
from multiprocessing import Pool
import pickle
import numpy as np
import codecs
import csv, string
import pycrfsuite, crf_utils
import random

def shuffle_docs_labels(docs, labels, filenames):
	random.seed(4)
	#zipped_data = zip(docs, labels, filenames)
	num_docs = len(docs)
	shuffle = random.sample(xrange(num_docs), num_docs)
	shuffled_docs = [docs[i] for i in shuffle]
	shuffled_labels = [labels[i] for i in shuffle]
	shuffled_filenames = [filenames[i] for i in shuffle]
	return shuffled_docs, shuffled_labels, shuffled_filenames

def shuffle_examples_preds(examples, preds):
	random.seed(4)
	#zipped_data = zip(examples, preds)
	num_docs = len(examples)
	shuffle = random.sample(xrange(num_docs), num_docs)
	shuffled_examples = [examples[i] for i in shuffle]
	shuffled_preds = [preds[i] for i in shuffle]
	return shuffled_examples, shuffled_preds

def create_base_w2v_dict(w2v_model, high_recall_wordlist):
	base_list = gen_utils.get_arg_wordlist()
	base_list += high_recall_wordlist.keys()
	base_dict = {}
	for word in base_list:
		if word not in base_dict and word in w2v_model:
			base_dict[word] = w2v_model[word]
	return base_dict

def get_w2v_dict(w2v_model, docs, high_recall_wordlist):
	print "Creating w2v dict"
	w2v_dict = create_base_w2v_dict(w2v_model, high_recall_wordlist)
	for doc in docs:
		for sentence in doc['sentences']:
			for token in sentence['tokens']:
				if token['word'] in w2v_dict or token['word'] not in w2v_model:
					continue
				w2v_dict[token['word']] = w2v_model[token['word']]
	return w2v_dict


def get_w2v_dict_args(w2v_model, examples, high_recall_wordlist):
	print "Creating w2v dict"
	w2v_dict = create_base_w2v_dict(w2v_model, high_recall_wordlist)
	for example in examples:
		sentence = example['annotated_sentence']
		#for sentence in doc['sentences']:
		for token in sentence['tokens']:
			if token['word'] in w2v_dict or token['word'] not in w2v_model:
				continue
			w2v_dict[token['word']] = w2v_model[token['word']]
	return w2v_dict	

extractor_pool = []

def create_parallel_infrastructure(docs, trigger_labels, filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, w2v_model, crf_file, realis=False):
	# need to create num_cores tfs, extractors, classifiers
	#tfs = [tf]
	#classifiers = [classifier]
	print "use_w2v is:"
	print use_w2v
	#global extractor_pool
	extractor_pool = []
	print 'loading w2v model'
	if not use_blank_w2v:
		print "Using real w2v"
		# use existing model
		#w2v_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(os.path.join(os.path.dirname("__file__"), '../../GoogleNews-vectors-negative300.bin'), binary=True)
	else:
		print 'using fake w2v'
		w2v_model = {}
	print 'w2v model loaded'
	start_time = time.time()
	docs, trigger_labels, filenames = shuffle_docs_labels(docs, trigger_labels, filenames)
	docs_per_core = int(math.ceil(float(len(docs)) / num_cores))
	docs_divided_among_cores = []
	trigger_labels_divided_among_cores = []
	filenames_divided_among_cores = []
	for i in xrange(num_cores):
		start_index = i * docs_per_core
		if i == num_cores - 1:
			end_index = len(docs)
		else:
			#end_index = min((i * docs_per_core + docs_per_core), len(docs))
			end_index = i * docs_per_core + docs_per_core
		docs_divided_among_cores.append(docs[start_index:end_index])
		trigger_labels_divided_among_cores.append(trigger_labels[start_index:end_index])
		filenames_divided_among_cores.append(filenames[start_index:end_index])	
	if len(extractor_pool) < num_cores:
		num_to_create = num_cores - len(extractor_pool)
		print "Duplicating extractors:"
		for i in xrange(num_to_create):
			#crf_tagger = pycrfsuite.Tagger()
			#crf_tagger.open(crf_file)	
			local_w2v_dict = get_w2v_dict(w2v_model, docs_divided_among_cores[i], high_recall_wordlist)
			token_featurizer = pattern_utils2.TokenFeaturizer(dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, local_w2v_dict, False)
			classifier = LogisticRegression()
			if realis:
				new_extractor = extraction_utils.RealisExtractor(dependency_type, reduce_deps, token_featurizer, classifier, True, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, crf_file, training=True)
			else:
				new_extractor = extraction_utils.EventExtractor(dependency_type, reduce_deps, token_featurizer, classifier, True, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, crf_file, training=True)
			#new_tf = deepcopy(tf)
			#new_classifier = deepcopy(classifier)
			#new_extractor = deepcopy(extractor)
			#tfs.append(new_tf)
			#classifiers.append(new_classifier)
			extractor_pool.append(new_extractor)
		# need to split docs into num_cores things
		print "Total time for duplication: " + str(float((time.time() - start_time)) / 60 ) + " minutes"
		print "Done duplicating extractors, shuffling docs..."
	extractors = extractor_pool[:num_cores]
	print "Docs shuffled..."
	#print "File order:"
	#print filenames_divided_among_cores
	return extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores

def create_parallel_infrastructure_args(examples, preds, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, w2v_model, crf_file):
	# need to create num_cores tfs, extractors, classifiers
	#tfs = [tf]
	#classifiers = [classifier]
	#global extractor_pool
	num_cores = num_cores * 10
	extractor_pool = []
	print 'loading w2v model'
	if not use_blank_w2v:
		print "Using real w2v"
		# do nothing
		#w2v_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(os.path.join(os.path.dirname("__file__"), '../../GoogleNews-vectors-negative300.bin'), binary=True)
	else:
		w2v_model = {}
	print 'w2v model loaded'
	start_time = time.time()
	examples, preds = shuffle_examples_preds(examples, preds)
	docs_per_core = int(math.ceil(float(len(examples)) / num_cores))
	examples_divided_among_cores = []
	preds_divided_among_cores = []	
	for i in xrange(num_cores):
		start_index = i * docs_per_core
		if i == num_cores - 1:
			end_index = len(examples)
		else:
			#end_index = min((i * docs_per_core + docs_per_core), len(docs))
			end_index = i * docs_per_core + docs_per_core
		examples_divided_among_cores.append(examples[start_index:end_index])
		preds_divided_among_cores.append(preds[start_index:end_index])
	if len(extractor_pool) < num_cores:
		num_to_create = num_cores - len(extractor_pool)
		print "Duplicating extractors:"
		for i in xrange(num_to_create):
			#crf_tagger = pycrfsuite.Tagger()
			#crf_tagger.open(crf_file)	
			local_w2v_dict = get_w2v_dict_args(w2v_model, examples_divided_among_cores[i], high_recall_wordlist)
			arg_featurizer = arg_utils.ArgFeaturizer(dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, local_w2v_dict, real_arg_deps, wn_types_features, False)	
			#	token_featurizer = pattern_utils2.TokenFeaturizer(dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, local_w2v_dict, False)
			classifier = LogisticRegression()
			new_extractor = extraction_utils.EventExtractor(dependency_type, reduce_deps, arg_featurizer, classifier, True, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, crf_file, training=True)
			#new_tf = deepcopy(tf)
			#new_classifier = deepcopy(classifier)
			#new_extractor = deepcopy(extractor)
			#tfs.append(new_tf)
			#classifiers.append(new_classifier)
			extractor_pool.append(new_extractor)
		# need to split docs into num_cores things
		print "Total time for duplication: " + str(float((time.time() - start_time)) / 60 ) + " minutes"
		print "Done duplicating extractors, shuffling docs..."
	extractors = extractor_pool[:num_cores]
	print "Docs shuffled..."
	#print "File order:"
	#print filenames_divided_among_cores
	return extractors, examples_divided_among_cores, preds_divided_among_cores



def create_parallel_infrastructure_eal(docs, filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, crf_file, w2v_model):
	# need to create num_cores tfs, extractors, classifiers
	#tfs = [tf]
	#classifiers = [classifier]
	#global extractor_pool
	extractor_pool = []
	#print 'loading w2v model'
	# if not use_blank_w2v:
	# 	#w2v_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(os.path.join(os.path.dirname("__file__"), '../../GoogleNews-vectors-negative300.bin'), binary=True)
	# else:
	# 	w2v_model = {}
	#print 'w2v model loaded'
	start_time = time.time()
	docs, filenames = shuffle_examples_preds(docs, filenames)
	docs_per_core = int(math.ceil(float(len(docs)) / num_cores))
	docs_divided_among_cores = []
	filenames_divided_among_cores = []
	w2v_model_dicts_divided_among_cores = []	
	for i in xrange(num_cores):
		start_index = i * docs_per_core
		if i == num_cores - 1:
			end_index = len(docs)
		else:
			#end_index = min((i * docs_per_core + docs_per_core), len(docs))
			end_index = i * docs_per_core + docs_per_core
		docs_divided_among_cores.append(docs[start_index:end_index])
		filenames_divided_among_cores.append(filenames[start_index:end_index])	
	if len(extractor_pool) < num_cores:
		num_to_create = num_cores - len(extractor_pool)
		print "Duplicating extractors:"
		for i in xrange(num_to_create):
			#crf_tagger = pycrfsuite.Tagger()
			#crf_tagger.open(crf_file)	
			local_w2v_dict = get_w2v_dict(w2v_model, docs_divided_among_cores[i], high_recall_wordlist)
			w2v_model_dicts_divided_among_cores.append(local_w2v_dict)
	print "Docs shuffled..."
	#print "File order:"
	#print filenames_divided_among_cores
	return docs_divided_among_cores, filenames_divided_among_cores, w2v_model_dicts_divided_among_cores



def parallel_process_docs(args):
	extractor = args[0]
	docs = args[1]
	trigger_labels = args[2]
	filenames = args[3]
	return extractor.process_docs(docs, trigger_labels, filenames)



def parallel_call_to_process_docs(docs, trigger_labels, filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, w2v_model, crf_file, realis=False):
	all_examples = []
	extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores = create_parallel_infrastructure(docs, trigger_labels, filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, w2v_model, crf_file, realis)
	#examples = Parallel(n_jobs = num_cores)(delayed(parallel_process_docs)(extractor, doc, triggers, doc_filename) for extractor, doc, triggers, doc_filename in zip(extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores))
	#p = Pool(num_cores)
	p = Pool(num_cores)
	examples = p.map(parallel_process_docs, ((extractor, doc, triggers, doc_filename) for extractor, doc, triggers, doc_filename in zip(extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores)))
	#examples = map(parallel_process_docs, ((extractor, doc, triggers, doc_filename) for extractor, doc, triggers, doc_filename in zip(extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores)))
	for example in examples:
		all_examples += example
	p.terminate()
	return all_examples



def consolidate_dicts(list_of_dicts):
	master_dict = {}
	for temp_dict in list_of_dicts:
		for key, val_list in temp_dict.items():
			if key in master_dict:
				master_dict[key] += val_list
			else:
				master_dict[key] = val_list
	return master_dict

def parallel_get_arg_features_dict(args):
	#print 'in parallel_get_arg_features_dict'
	extractor = args[0]
	examples = args[1]
	preds = args[2]
	dependency_type = args[3]
	reduce_deps = args[4]
	crf_file = args[5]
	arg_patterns = []
	noun_patterns = []
	person_patterns = []
	event_features_dict = {}
	num_examples = len(examples)
	crf_tagger = pycrfsuite.Tagger()
	crf_tagger.open(crf_file)
	#print "num examples is ", str(num_examples)	
	count_examples = 0
	for i, pred in enumerate(preds):
		#print i
		pred = str(pred)
		#if i % 100 == 0:
		#	print 'gathering args... ' + str(float(i) / num_examples)
		if pred == "NONE":
			continue
		example = examples[i]
		trigger_token = example['token']
		sentence = example['annotated_sentence']
		triggers = example['token_triggers']
		event_word_counts = example['event_word_counts']
		#hashed_sentence = list(extractor.sentence_hasher.hash_annotated_sentence(sentence))
		hashed_sentence = []
		crf_features, _ = crf_utils.process_sentence(sentence, [])
		entity_tagged_sentence = crf_tagger.tag(crf_features)
		entities = gen_utils.get_entities(entity_tagged_sentence, sentence, dependency_type)	
		for trigger in triggers:
			#event_type = trigger['ere_type'] + "." + trigger['subtype']
			event_type = trigger['event_type']
			if event_type != pred:
				print event_type, " not equal to ", pred
				continue
			arg_roles = trigger['arg_roles']
			arg_offsets = trigger['args']
			arg_types = trigger['arg_types']
			#role_tokens = []
			found_entities = map(lambda x: False, entities)
			for j, role in enumerate(arg_roles):
				arg_type = arg_types[j]
				offset = arg_offsets[j]
				arg_tokens = gen_utils.get_tokens_from_offsets(offset, sentence['tokens'])
				#role_tokens.append(arg_tokens)
				role_head = gen_utils.get_head(arg_tokens, sentence['tokens'], sentence[dependency_type])
				if not role_head:
					#print "couldn't find head for:"
					#print role
					#print arg_tokens
					#print offset
					continue
				token_to_classify = role_head
				for k, entity in enumerate(entities):
					#found_entities[k] = False
					for entity_token in entity['tokens']:
						if entity_token in arg_tokens:
							found_entities[k] = True
							token_to_classify = entity['head']
							entity_type = entity['type']
				if token_to_classify == role_head:
					entity_type = arg_type
				token = token_to_classify
				token_back_bigram = extractor.tf.get_token_back_bigram(token, sentence)
				token_forward_bigram = extractor.tf.get_token_forward_bigram(token, sentence)
				#hashed_sentence = list(extractor.sentence_hasher.hash_local_context(token, sentence))
				#paths, _ = gen_utils.get_arg_paths_plus_wn_types(trigger_token, token, sentence, reduce_deps, dependency_type, False)
				path = gen_utils.find_shortest_path_to_token(trigger_token, sentence[dependency_type], sentence['tokens'], None, ['ROOT', 'root'], 'both', token, reduce_deps)
				paths = [path]
				features = extractor.tf.construct_features(token, trigger_token, sentence, extractor.event_dict.keys(), event_word_counts, arg_patterns, noun_patterns, person_patterns, paths, token_back_bigram, token_forward_bigram, hashed_sentence, entity_type, entity_tagged_sentence)
				token_features_label = {}
				token_features_label['event'] = event_type
				token_features_label['features'] = features
				token_features_label['role'] = role
				count_examples += 1
				if event_type in event_features_dict:
					event_features_dict[event_type].append(token_features_label)
				else:
					event_features_dict[event_type] = [token_features_label]
			for j, entity in enumerate(entities):
				found = found_entities[j]
				if not found:
					token = entity['head']
					role = 'NONE'
					entity_type = entity['type']
					token_back_bigram = extractor.tf.get_token_back_bigram(token, sentence)
					token_forward_bigram = extractor.tf.get_token_forward_bigram(token, sentence)
					#hashed_sentence = list(extractor.sentence_hasher.hash_local_context(token, sentence))
					#paths, _ = gen_utils.get_arg_paths_plus_wn_types(trigger_token, token, sentence, reduce_deps, dependency_type, False)
					path = gen_utils.find_shortest_path_to_token(trigger_token, sentence[dependency_type], sentence['tokens'], None, ['ROOT', 'root'], 'both', token, reduce_deps)
					paths = [path]
					features = extractor.tf.construct_features(token, trigger_token, sentence, extractor.event_dict.keys(), event_word_counts, arg_patterns, noun_patterns, person_patterns, paths, token_back_bigram, token_forward_bigram, hashed_sentence, entity_type, entity_tagged_sentence)
					token_features_label = {}
					token_features_label['event'] = event_type
					token_features_label['features'] = features
					token_features_label['role'] = role
					count_examples += 1
					if event_type in event_features_dict:
						event_features_dict[event_type].append(token_features_label)
					else:
						event_features_dict[event_type] = [token_features_label]
	#print 'finished parallel_get_arg_features_dict'
	#print 'num examples created:', str(count_examples)		
	#examples = None
	#extractor = None
	return event_features_dict

def get_event_id(tokens, event):
	token_word_list = map(lambda x: x['word'], tokens)
	token_string = ""
	for word in token_word_list:
		token_string += word
	token_string += event
	token_string += str(tokens[-1]['index'])
	return generate_line_id(token_string)


def make_sentence_smash(sentence):
	smash = ""
	for token in sentence['tokens']:
		smash += token['word']
	return smash

def count_triggers_per_sentence(preds, examples):
	smash_dict = {}
	for i, pred in enumerate(preds):
		#print i
		pred = str(pred)
		#if i % 100 == 0:
		#	print 'gathering args... ' + str(float(i) / num_examples)
		if pred == "NONE":
			continue
		example = examples[i]
		#confidence = confidences[i]
		#trigger_token = example['token']
		sentence = example['annotated_sentence']
		smash = make_sentence_smash(sentence)
		if smash in smash_dict:
			smash_dict[smash] += 1
		else:
			smash_dict[smash] = 1
	return smash_dict

def parallel_get_arg_features_dict_eal(args):
	#print 'in parallel_get_arg_features_dict'
	extractor = args[0]
	examples = args[1]
	preds = args[2]
	confidences = args[3]
	dependency_type = args[4]
	reduce_deps = args[5]
	realis_classifier = args[6]
	realis_featurizer = args[7]
	crf_tagger = args[8]
	arg_patterns = []
	noun_patterns = []
	person_patterns = []
	event_features_dict = {}
	num_examples = len(examples)
	smash_dict = count_triggers_per_sentence(preds, examples)
	for i, pred in enumerate(preds):
		#print i
		pred = str(pred)
		#if i % 100 == 0:
		#	print 'gathering args... ' + str(float(i) / num_examples)
		if pred == "NONE":
			continue
		example = examples[i]
		confidence = confidences[i]
		trigger_token = example['token']
		sentence = example['annotated_sentence']
		event_word_counts = example['event_word_counts']
		smash = make_sentence_smash(sentence)
		sentence_length = len(sentence['tokens'])
		percent_triggers = float(smash_dict[smash]) / sentence_length
		if (smash_dict[smash] > 5 and percent_triggers > 0.2) or smash_dict[smash] > 15 or percent_triggers > 0.34:
			print 'smash dict is ', str(smash_dict[smash]), str(percent_triggers)
			print smash, pred, trigger_token['word']
			continue
		event_id = get_event_id(example['annotated_sentence']['tokens'] + [example['token']], pred)
		#hashed_sentence = list(extractor.sentence_hasher.hash_annotated_sentence(sentence))
		hashed_sentence = []
		#realis_features = example['token_features'] + hashed_sentence
		#realis_features = hashed_sentence
		realis_features, _ = realis_featurizer.process_token(trigger_token, sentence, [], event_word_counts, hashed_sentence)
		#print realis_features
		realis = realis_classifier.predict([realis_features])[0]
		#print "REALIS:", realis
		crf_features, _ = crf_utils.process_sentence(sentence, [])
		entity_tagged_sentence = crf_tagger.tag(crf_features)
		#print entity_tagged_sentence
		entities = gen_utils.get_entities(entity_tagged_sentence, sentence, dependency_type)
		total_entity_tokens = 0
		for entity in entities:
			total_entity_tokens += len(entity['tokens'])
		percent_entity_tokens = float(total_entity_tokens) / sentence_length
		percent_entities = float(len(entities)) / sentence_length
		if len(entities) > 20 or percent_entities > 0.5 or percent_entity_tokens > 0.8:
			print 'Found list, too many entities in sentence, skipping,', str(len(entities)), str(percent_entities), str(percent_entity_tokens)
			continue
		#print len(entities)
		#for token in sentence['tokens']:
		for entity in entities:
			token = entity['head']
			entity_tokens = entity['tokens']
			entity_type = entity['type']
			entity_offsets = [int(entity_tokens[0]['characterOffsetBegin']), int(entity_tokens[-1]['characterOffsetEnd'])]
			s = gen_utils.create_string_from_tokens(sentence['tokens'])
			s = filter(lambda x: x in string.printable, s)
			token_back_bigram = extractor.tf.get_token_back_bigram(token, sentence)
			token_forward_bigram = extractor.tf.get_token_forward_bigram(token, sentence)
			#paths, _ = gen_utils.get_arg_paths_plus_wn_types(trigger_token, token, sentence, reduce_deps, dependency_type, False)
			path = gen_utils.find_shortest_path_to_token(trigger_token, sentence[dependency_type], sentence['tokens'], None, ['ROOT', 'root'], 'both', token, reduce_deps)
			paths = [path]
			features = extractor.tf.construct_features(token, trigger_token, sentence, extractor.event_dict.keys(), event_word_counts, arg_patterns, noun_patterns, person_patterns, paths, token_back_bigram, token_forward_bigram, hashed_sentence, entity_type, entity_tagged_sentence)
			token_features_label = {}
			token_features_label['event'] = pred
			token_features_label['features'] = features
			token_features_label['trigger_offsets'] = [trigger_token['characterOffsetBegin'], int(trigger_token['characterOffsetEnd']) - 1]
			token_features_label['arg_offsets'] = [token['characterOffsetBegin'], int(token['characterOffsetEnd']) - 1]
			token_features_label['arg_cas'] = token['word']
			token_features_label['trigger_confidence'] = confidence
			token_features_label['event_id'] = event_id
			token_features_label['realis'] = realis
			token_features_label['token_index'] = token['index']
			token_features_label['sentence_string'] = s
			if pred in event_features_dict:
				event_features_dict[pred].append(token_features_label)
			else:
				event_features_dict[pred] = [token_features_label]
	return event_features_dict

def parallel_call_to_process_args(examples, preds, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, w2v_model, crf_file):
	extractors, examples_divided_among_cores, preds_divided_among_cores = create_parallel_infrastructure_args(examples, preds, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, w2v_model, crf_file)
	p = Pool(num_cores)
	event_feature_dict_list = p.map(parallel_get_arg_features_dict, ((extractor, examples, preds, dependency_type, reduce_deps, crf_file) for extractor, examples, preds in zip(extractors, examples_divided_among_cores, preds_divided_among_cores)))
	event_feature_dict = consolidate_dicts(event_feature_dict_list)
	p.terminate()
	return event_feature_dict

def train_classifier(examples):
	features = map(lambda x: x['features'], examples)
	labels = map(lambda x: x['role'], examples)
	if len(set(labels)) < 2:
		return False
	lr = LogisticRegression()
	lr.fit(features, labels)
	return lr	

def parallel_train_arg_classifiers(event_arg_features_dict, num_cores):
	events = event_arg_features_dict.keys()
	example_lists = event_arg_features_dict.values()
	#classifiers = Parallel(n_jobs = num_cores)(delayed(train_classifier)(example_list) for example_list in example_lists)
	p = Pool(num_cores)
	classifiers = p.map(train_classifier, (example_list for example_list in example_lists))
	classifier_dict = {}
	for i, event in enumerate(events):
		if classifiers[i] == False:
			print event, ': No args to train on'
			continue
		classifier_dict[event] = classifiers[i]
	p.terminate()
	return classifier_dict


def parallel_extract_from_docs_and_save(args):
	extractor = args[0]
	docs = args[1]
	filenames = args[2]
	save_dir = args[3]
	extractor.extract_from_docs_and_save(docs, filenames, save_dir)

def parallel_call_to_extract_from_docs_and_save(extractor, docs, filenames, save_dir, num_cores=1):
	empty_trigger_labels = map(lambda x: [], xrange(len(docs)))
	extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores = create_parallel_infrastructure(extractor, docs, empty_trigger_labels, filenames, num_cores)	
	print "Beginning parallel processing... "
	p = Pool(num_cores)
	p.map(parallel_extract_from_docs_and_save, ((extractor, docs, doc_filenames, save_dir) for extractor, docs, doc_filenames in zip(extractors, docs_divided_among_cores, filenames_divided_among_cores)))
	p.terminate()

def generate_line_id(output_string):
	return str(abs(hash(output_string)))


def get_result_line(event, confidence, filename, raw_text, trigger_offsets, realis):
	system_id = "UWashington400"
	char_span = str(trigger_offsets[0]) + ',' + str(int(trigger_offsets[1]) + 1)
	mention = raw_text[int(trigger_offsets[0]):int(trigger_offsets[1] + 1)]
	mention_id = generate_line_id(filename + event + str(confidence) + raw_text + char_span)
	result_line = str(system_id) + '\t' + filename + '\t' + str(mention_id) + '\t' + char_span + '\t' + mention + '\t' + event + '\t' + realis + '\t' + str(confidence) + '\t' + str(confidence) + '\t' + str(confidence)
	return result_line


def process_save_doc(doc, filename, trigger_extractor, arg_extractor, trigger_classifier, arg_classifier_dict, save_dir, dependency_type, reduce_deps, realis_classifier, raw_text_directory, realis_featurizer, tagger):
	print "in doc", filename
	logfile = 'log.txt'
	if actual_run:
		if 'ENG_DF' in filename:
			doc_directory = raw_text_directory + 'df/'
		else:
			doc_directory = raw_text_directory + 'nw/'
		raw_text_filename = doc_directory + filename
		with codecs.open(raw_text_filename, encoding='utf-8') as f:
			raw_text = f.read()		
		filename = filename.split('.txt')[0]	
	else:
		if '.mpdf.xml' in filename:
			doc_directory = raw_text_directory #+ ''
			raw_text_filename = doc_directory + filename
			with codecs.open(raw_text_filename, encoding='utf-8') as f:
				raw_text = f.read()
			filename = filename.split('.mpdf.xml')[0]
		elif '.txt' in filename:
			doc_directory = raw_text_directory # + ''
			raw_text_filename = doc_directory + filename
			with codecs.open(raw_text_filename, encoding='utf-8') as f:
				raw_text = f.read()		
			filename = filename.split('.txt')[0]
		else:
			filename = filename + '.txt'
			doc_directory = raw_text_directory # + ''
			raw_text_filename = doc_directory + filename
			with codecs.open(raw_text_filename, encoding='utf-8') as f:
				raw_text = f.read()		
			filename = filename.split('.txt')[0]			 
	write_filename = save_dir + 'nuggets_all_training_all_words_actual3.tbf'
	lines_to_write = []
	sentences_to_write = []
	examples = trigger_extractor.process_doc(doc, [], filename)	
	trigger_features, labels, sentences, words, labeled_triggers = extraction_utils.get_corpus_info(examples)
	events = trigger_classifier.predict(trigger_features)
	confidences = map(lambda x: 1, events)
	for pred_index, event in enumerate(events):
		if event == "NONE":
			continue
		confidence = confidences[pred_index]
		assert len(confidences) == len(examples)
		example = examples[pred_index]
		trigger_token = example['token']
		sentence = example['annotated_sentence']
		event_word_counts = example['event_word_counts']
		realis_features, _ = realis_featurizer.process_token(trigger_token, sentence, [], event_word_counts, [])
		realis = realis_classifier.predict([realis_features])[0]				
		trigger_offsets = [trigger_token['characterOffsetBegin'], int(trigger_token['characterOffsetEnd']) - 1]
		result_line = get_result_line(event, confidence, filename, raw_text, trigger_offsets, realis) 
		lines_to_write.append(result_line)
	if len(lines_to_write) > 250:
		print "too many extractions, skipping file:", str(len(lines_to_write)), filename
		lines_to_write = []
		event_linking_dict = {}
	with codecs.open(write_filename, 'a', encoding='utf-8') as f:
		f.write('#BeginOfDocument ' + filename + '\n')
		for item in lines_to_write:
			f.write(item + '\n')
		f.write('#EndOfDocument' + '\n')
		f.flush()




def parallel_extract_from_docs_and_save_with_args(args):
	docs = args[0]
	filenames = args[1]
	w2v_model = args[2]
	save_dir = args[3]
	dependency_type = args[4]
	reduce_deps = args[5]
	use_lemmas = args[6]
	use_w2v = args[7]
	noms_filename = args[8]
	arg_deps = args[9]
	noun_deps = args[10]
	person_deps = args[11]
	bigram_set = args[12]
	high_recall_wordlist = args[13]
	event_dict = args[14]
	check_only_nouns = args[15]
	check_all_noms = args[16]
	real_arg_deps = args[17]
	wn_types_features = args[18]
	trigger_classifier_filename = args[19]
	classifier_dict_filename = args[20]
	realis_classifier_filename = args[21]
	raw_text_directory = args[22]
	crf_file = args[23]
	num_cores = 1
	with open(trigger_classifier_filename, 'rb') as f:
		trigger_classifier = pickle.load(f)
	with open(classifier_dict_filename, 'rb') as f:
		arg_classifier_dict = pickle.load(f)
	with open(realis_classifier_filename, 'rb') as f:
		realis_classifier = pickle.load(f)	
	tagger = pycrfsuite.Tagger()
	tagger.open(crf_file)
	token_featurizer = pattern_utils2.TokenFeaturizer(dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, w2v_model, False)
	classifier = False
	trigger_extractor = extraction_utils.EventExtractor(dependency_type, reduce_deps, token_featurizer, classifier, True, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, crf_file, training=True)
	person_deps = [] # TODO
	arg_featurizer = arg_utils.ArgFeaturizer(dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, w2v_model, real_arg_deps, wn_types_features, False)	
	arg_extractor = extraction_utils.EventExtractor(dependency_type, reduce_deps, arg_featurizer, classifier, True, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, crf_file, training=True)
	realis_featurizer = extraction_utils.RealisExtractor(dependency_type, reduce_deps, arg_featurizer, classifier, True, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, crf_file, training=False)
	for i, doc in enumerate(docs):
		filename = filenames[i]
		process_save_doc(doc, filename, trigger_extractor, arg_extractor, trigger_classifier, arg_classifier_dict, save_dir, dependency_type, reduce_deps, realis_classifier, raw_text_directory, realis_featurizer, tagger)
	# create tf
	# create trigger extractor
	# create af
	# create arg extractor
	# for each doc:
	#   process doc with trigger extractor
	#   classify features
	#   for each positive trigger:
	#     process sentence with arg extractor
	#     classify features
	#     for each positive arg:
	#       save results




def parallel_call_to_extract_from_docs_and_save_with_args(docs, filenames, save_dir, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, trigger_classifier_filename, classifier_dict_filename, realis_classifier_filename, raw_text_directory, crf_file, w2v_model):
	print dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, check_only_nouns, check_all_noms, trigger_classifier_filename, classifier_dict_filename, realis_classifier_filename, raw_text_directory, crf_file
	empty_trigger_labels = map(lambda x: [], xrange(len(docs)))
	docs_divided_among_cores, filenames_divided_among_cores, w2v_model_dicts_divided_among_cores = create_parallel_infrastructure_eal(docs, filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, crf_file, w2v_model)	
	p = Pool(num_cores)
	#p.map(parallel_extract_from_docs_and_save_with_args, ((docs, filenames, w2v_model_dicts, save_dir, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, trigger_classifier_filename, classifier_dict_filename, realis_classifier_filename, raw_text_directory, crf_file) for docs, filenames, w2v_model_dicts in zip(docs_divided_among_cores, filenames_divided_among_cores, w2v_model_dicts_divided_among_cores)))
	map(parallel_extract_from_docs_and_save_with_args, ((docs, filenames, w2v_model_dicts, save_dir, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, trigger_classifier_filename, classifier_dict_filename, realis_classifier_filename, raw_text_directory, crf_file) for docs, filenames, w2v_model_dicts in zip(docs_divided_among_cores, filenames_divided_among_cores, w2v_model_dicts_divided_among_cores)))
	p.terminate()

#p = []
def set_pool(num_cores):
	global p
	p = Pool(num_cores)

use_blank_w2v = False
actual_run = False