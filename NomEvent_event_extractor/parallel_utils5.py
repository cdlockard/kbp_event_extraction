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
	#num_cores = num_cores * 10
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



def create_parallel_infrastructure_eal(docs, filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, crf_file):
	# need to create num_cores tfs, extractors, classifiers
	#tfs = [tf]
	#classifiers = [classifier]
	#global extractor_pool
	extractor_pool = []
	print 'loading w2v model'
	if not use_blank_w2v:
		w2v_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(os.path.join(os.path.dirname("__file__"), '../../GoogleNews-vectors-negative300.bin'), binary=True)
	else:
		w2v_model = {}
	print 'w2v model loaded'
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
			#token_featurizer = pattern_utils2.TokenFeaturizer(dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, local_w2v_dict, False)
			#classifier = LogisticRegression()
			#new_trigger_extractor = extraction_utils.EventExtractor(dependency_type, reduce_deps, token_featurizer, classifier, True, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, training=True)
			#new_tf = deepcopy(tf)
			#new_classifier = deepcopy(classifier)
			#new_extractor = deepcopy(extractor)
			#tfs.append(new_tf)
			#classifiers.append(new_classifier)
			#extractor_pool.append(new_extractor)
		# need to split docs into num_cores things
		#print "Total time for duplication: " + str(float((time.time() - start_time)) / 60 ) + " minutes"
		#print "Done duplicating extractors, shuffling docs..."
	#extractors = extractor_pool[:num_cores]
	print "Docs shuffled..."
	#print "File order:"
	#print filenames_divided_among_cores
	return docs_divided_among_cores, filenames_divided_among_cores, w2v_model_dicts_divided_among_cores



#def parallel_process_docs(extractor, docs, trigger_labels, filenames):
#	return extractor.process_docs(docs, trigger_labels, filenames)

def parallel_process_docs(args):
	#global examples
	extractor = args[0]
	docs = args[1]
	trigger_labels = args[2]
	filenames = args[3]
	#examples += extractor.process_docs(docs, trigger_labels, filenames)
	return extractor.process_docs(docs, trigger_labels, filenames)
	

# def parallel_call_to_process_docs(extractor, docs, trigger_labels, filenames, num_cores):
# 	all_examples = []
# 	extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores = create_parallel_infrastructure(extractor, docs, trigger_labels, filenames, num_cores)
# 	examples = Parallel(n_jobs = num_cores)(delayed(parallel_process_docs)(extractor, doc, triggers, doc_filename) for extractor, doc, triggers, doc_filename in zip(extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores))
# 	for example in examples:
# 		all_examples += example
# 	return all_examples

examples = False

def parallel_call_to_process_docs(docs, trigger_labels, filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, w2v_model, crf_file, realis=False):
	all_examples = []	
	extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores = create_parallel_infrastructure(docs, trigger_labels, filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, w2v_model, crf_file, realis)
	#examples = Parallel(n_jobs = num_cores)(delayed(parallel_process_docs)(extractor, doc, triggers, doc_filename) for extractor, doc, triggers, doc_filename in zip(extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores))
	#p = Pool(num_cores)
	examples = map(parallel_process_docs, ((extractor, doc, triggers, doc_filename) for extractor, doc, triggers, doc_filename in zip(extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores)))	
	for example in examples:
		all_examples += example
	# with multiprocessing.Manager() as manager:
	# 	global examples
	# 	examples = manager.list()
	# 	p = Pool(num_cores)
	# 	#examples = p.map(parallel_process_docs, ((extractor, doc, triggers, doc_filename) for extractor, doc, triggers, doc_filename in zip(extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores)))
	# 	null_list = p.map(parallel_process_docs, ((extractor, doc, triggers, doc_filename) for extractor, doc, triggers, doc_filename in zip(extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores)))
	# 	#examples = map(parallel_process_docs, ((extractor, doc, triggers, doc_filename) for extractor, doc, triggers, doc_filename in zip(extractors, docs_divided_among_cores, trigger_labels_divided_among_cores, filenames_divided_among_cores)))
	# 	#for example in examples:
	# 	#	all_examples += example
	# 	all_examples = list(examples)
	# 	p.terminate()
	# return all_examples
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
		# for trigger in triggers:
		# 	#event_type = trigger['ere_type'] + "." + trigger['subtype']
		# 	event_type = trigger['event_type']
		# 	if event_type != pred:
		# 		print event_type, " not equal to ", pred
		# 		continue
		# 	arg_roles = trigger['arg_roles']
		# 	arg_offsets = trigger['args']
		# 	role_tokens = []
		# 	for j, arg in enumerate(arg_roles):
		# 		offset = arg_offsets[j]
		# 		arg_tokens = gen_utils.get_tokens_from_offsets(offset, sentence['tokens'])
		# 		role_tokens.append(arg_tokens)
		# 	#for token in sentence['tokens']:	
		# 	#for token in map(lambda x: x['head'], entities):
		# 	for entity in entities:
		# 		token = entity['head']
		# 		token_role = "NONE"
		# 		entity_type = entity['type']
		# 		for j, role in enumerate(arg_roles):
		# 			arg_tokens = role_tokens[j]
		# 			#if token in arg_tokens:
		# 			#	token_role = role
		# 			for entity_token in entity['tokens']:
		# 				if entity_token in arg_tokens:
		# 					token_role = role
		# 		token_back_bigram = extractor.tf.get_token_back_bigram(token, sentence)
		# 		token_forward_bigram = extractor.tf.get_token_forward_bigram(token, sentence)
		# 		#hashed_sentence = list(extractor.sentence_hasher.hash_local_context(token, sentence))
		# 		#paths, _ = gen_utils.get_arg_paths_plus_wn_types(trigger_token, token, sentence, reduce_deps, dependency_type, False)
		# 		path = gen_utils.find_shortest_path_to_token(trigger_token, sentence[dependency_type], sentence['tokens'], None, ['ROOT', 'root'], 'both', token, reduce_deps)
		# 		paths = [path]
		# 		features = extractor.tf.construct_features(token, trigger_token, sentence, extractor.event_dict.keys(), event_word_counts, arg_patterns, noun_patterns, person_patterns, paths, token_back_bigram, token_forward_bigram, hashed_sentence, entity_type, entity_tagged_sentence)
		# 		token_features_label = {}
		# 		token_features_label['event'] = event_type
		# 		token_features_label['features'] = features
		# 		token_features_label['role'] = token_role
		# 		count_examples += 1
		# 		if event_type in event_features_dict:
		# 			event_features_dict[event_type].append(token_features_label)
		# 		else:
		# 			event_features_dict[event_type] = [token_features_label]		
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
					# if token_to_classify['ner'] == "PERSON":
					# 	entity_type = "PER"
					# elif token_to_classify['ner'] == "ORGANIZATION":
					# 	entity_type = 'ORG'
					# elif token_to_classify['ner'] == "LOCATION":
					# 	entity_type = 'LOC'
					# else:				
					# 	possible_roles = gen_utils.event_role_type_dict[event_type][role]		
					# 	entity_type = possible_roles[random.randint(0, len(possible_roles) - 1)]
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
		sentence_length = len(sentence['tokens'])
		total_entity_tokens = 0
		for entity in entities:
			total_entity_tokens += len(entity['tokens'])
		percent_entity_tokens = float(total_entity_tokens) / sentence_length
		percent_entities = float(len(entities)) / sentence_length
		#print str(len(entities)), str(percent_entities), str(percent_entity_tokens)
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
	#global p
	extractors, examples_divided_among_cores, preds_divided_among_cores = create_parallel_infrastructure_args(examples, preds, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, w2v_model, crf_file)
	#event_feature_dicts = Parallel(n_jobs = num_cores)(delayed(parallel_get_arg_features_dict)(extractor, examples, preds) for extractor, examples, preds in zip(extractors, examples_divided_among_cores, preds_divided_among_cores))
	##p = Pool(num_cores)
	#tagger = pycrfsuite.Tagger()
	#tagger.open(crf_file)
	p = Pool(num_cores)
	#event_feature_dict_list = p.map(parallel_get_arg_features_dict, ((extractor, examples, preds, dependency_type, reduce_deps, crf_file) for extractor, examples, preds in zip(extractors, examples_divided_among_cores, preds_divided_among_cores)))
	event_feature_dict_list = map(parallel_get_arg_features_dict, ((extractor, examples, preds, dependency_type, reduce_deps, crf_file) for extractor, examples, preds in zip(extractors, examples_divided_among_cores, preds_divided_among_cores)))
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
	#examples = Parallel(n_jobs = num_cores)(delayed(parallel_extract_from_docs_and_save)(extractor, docs, doc_filenames, save_dir) for extractor, docs, doc_filenames in zip(extractors, docs_divided_among_cores, filenames_divided_among_cores))
	p = Pool(num_cores)
	p.map(parallel_extract_from_docs_and_save, ((extractor, docs, doc_filenames, save_dir) for extractor, docs, doc_filenames in zip(extractors, docs_divided_among_cores, filenames_divided_among_cores)))
	p.terminate()

def generate_line_id(output_string):
	return str(abs(hash(output_string)))

def get_result_line(role, features, arg_confidence, filename, raw_text):
	event = features['event']
	trigger_offsets = features['trigger_offsets']
	arg_offset = features['arg_offsets']
	cas_offset = str(arg_offset[0]) + "-" + str(arg_offset[1])
	#cas = features['arg_cas']
	cas = raw_text[int(arg_offset[0]):int(arg_offset[1] + 1)]
	cas = cas.replace('\n', ' ')
	cas = cas.replace('\t', ' ')
	trigger_confidence = features['trigger_confidence']
	realis = features['realis']
	predicate_justification_offset = str(min(int(trigger_offsets[0]), int(arg_offset[0]))) + "-" + str(max(int(trigger_offsets[1]), int(arg_offset[1])))
	confidence_to_write = trigger_confidence * arg_confidence
	additional_arg_justification = "NIL"
	#realis = "ACTUAL"
	base_filler_offset = cas_offset
	output_string = filename + '\t' + event + '\t' + role + '\t' + cas + '\t' + cas_offset + '\t' + predicate_justification_offset + '\t' + base_filler_offset + '\t' + additional_arg_justification + '\t' + realis + '\t' + str(confidence_to_write)
	line_id = generate_line_id(output_string)
	output_string = line_id + '\t' + output_string + '\n'
	return output_string, line_id


def process_save_doc(doc, filename, trigger_extractor, arg_extractor, trigger_classifier, arg_classifier_dict, save_dir, dependency_type, reduce_deps, realis_classifier, raw_text_directory, realis_featurizer, tagger):
	print "in doc", filename
	logfile = 'log.txt'
	if '.mpdf.xml' in filename:
		doc_directory = raw_text_directory #+ ''
		raw_text_filename = doc_directory + filename
		with codecs.open(raw_text_filename, encoding='utf-8') as f:
			raw_text = f.read()
		filename = filename.split('.mpdf.xml')[0]
	elif '.xml' in filename:
		doc_directory = raw_text_directory # + ''
		raw_text_filename = doc_directory + filename
		#with open(raw_text_filename, 'rb') as f:
		with codecs.open(raw_text_filename, encoding='utf-8') as f:
			raw_text = f.read()		
		filename = filename.split('.xml')[0]
	write_filename = save_dir + 'arguments/' + filename
	linking_filename = save_dir + 'linking/' + filename
	lines_to_write = []
	sentences_to_write = []
	examples = trigger_extractor.process_doc(doc, [], filename)
	if len(examples) == 0:
		print '0 feature document', filename
		with codecs.open(write_filename, 'wb', encoding='utf-8') as f:
			for item in []:
				f.write(item)
		with open(linking_filename, 'wb') as f:				
			for item in []:
				f.write(item)
		with codecs.open(logfile , 'a', encoding='utf-8') as f:
			f.write(write_filename + ', ' + str(len(lines_to_write)) + '\n')
		return		
	trigger_features, labels, sentences, words, labeled_triggers = extraction_utils.get_corpus_info(examples)
	#print len(trigger_features)
	#print len(trigger_features[0])
	#pred = trigger_classifier.predict(examples)
	pred = trigger_classifier.predict_proba(trigger_features)
	confidences = np.max(pred, 1)
	event_indices = np.argmax(pred, 1)
	events = map(lambda x: str(trigger_classifier.classes_[x]), event_indices)
	# comment back in TODO
	# for pred_index, event in enumerate(events):
	# 	if event == "NONE":
	# 		lr_confidences = pred[pred_index]
	# 		second_most = lr_confidences.argsort()[1]
	# 		second_most_val = lr_confidences[second_most]
	# 		#print second_most_val
	# 		if second_most_val > .2:
	# 			event = str(trigger_classifier.classes_[second_most])
	# 			print 'adding event ', event, second_most_val
	# 			events[pred_index] = event
	# 			confidences[pred_index] = second_most_val
	# 		else:
	# 			continue			
	arg_features_dict = parallel_get_arg_features_dict_eal((arg_extractor, examples, events, confidences, dependency_type, reduce_deps, realis_classifier, realis_featurizer, tagger))
	event_linking_dict = {}
	for pred_index, [event, features_list] in enumerate(arg_features_dict.items()):
		if event == "NONE":
			continue
		#print 'in event', event
		#print len(features_list)
		#print len(features_list[0])
		arg_features_list = map(lambda x: x['features'], features_list)
		if event not in arg_classifier_dict:
			print "NO ARG CLASSIFIER FOR ", event
			continue
		arg_pred = arg_classifier_dict[event].predict_proba(arg_features_list)
		arg_confidences = np.max(arg_pred, 1)
		role_indices = np.argmax(arg_pred, 1)
		roles = map(lambda x: str(arg_classifier_dict[event].classes_[x]), role_indices)
		# TODO: comment back in:
		# for pred_index, role in enumerate(roles):
		# 	if role == "NONE":
		# 		lr_confidences = arg_pred[pred_index]
		# 		second_most = lr_confidences.argsort()[1]
		# 		second_most_val = lr_confidences[second_most]
		# 		#print second_most_val
		# 		if second_most_val > .1:
		# 			role = str(arg_classifier_dict[event].classes_[second_most])
		# 			print 'adding role ', role, second_most_val
		# 			print filename
		# 			roles[pred_index] = role
		# 			arg_confidences[pred_index] = second_most_val
		# 		else:
		# 			continue			
		for i, role in enumerate(roles):
			if role == "NONE":
				continue
			arg_confidence = arg_confidences[i]
			features = features_list[i]
			# if features['arg_cas'].lower() in ['the', 'a', 'an']:
			# 	continue
			# if i >= 1 and features['token_index'] > 1:
			# 	prev_features = features_list[i - 1]
			# 	if features['event_id'] == prev_features['event_id']:# and features['token_index'] == (prev_features['token_index'] + 1):
			# 		if roles[i-1] == role or features['arg_cas'] in ["'s", '"', '.', ',', "'", '?', 'of', 'in']:
			# 			#prev_features = features_list[i - 1]
			# 			prev_offset = prev_features['arg_offsets']
			# 			prev_string = prev_features['arg_cas']
			# 			num_spaces = int(features['arg_offsets'][0]) - int(prev_offset[1]) - 1
			# 			space_filler = ""
			# 			for j in xrange(num_spaces):
			# 				space_filler += " "
			# 			features['arg_offsets'][0] = prev_offset[0]
			# 			features['arg_cas'] = prev_string + space_filler + features['arg_cas']
			# 		#elif features['arg_cas'] in 
			# 		if (roles[i-1] == role) and prev_string.lower() not in ['the', 'a', 'an']:
			# 			lines_to_write = lines_to_write[:-1]
			# 			sentences_to_write = sentences_to_write[:-1]
			# 			if prev_features['realis'] != 'Generic':
			# 				event_linking_dict[features['event_id']] = event_linking_dict[features['event_id']][:-1]
			full_confidence = arg_confidence * features['trigger_confidence']
			if full_confidence < 0.85:
				continue
			result_line, line_id = get_result_line(role, features, arg_confidence, filename, raw_text) 
			lines_to_write.append(result_line)
			s = features['sentence_string']
			sentences_to_write.append(s)
			#print result_line
			if features['realis'] == 'Generic':
				continue
			if features['event_id'] in event_linking_dict:
				event_linking_dict[features['event_id']].append(line_id)
			else:
				event_linking_dict[features['event_id']] = [line_id]
	if len(lines_to_write) > 250:
		print "too many extractions, skipping file:", str(len(lines_to_write)), filename
		lines_to_write = []
		event_linking_dict = {}			
	with codecs.open(write_filename, 'wb', encoding='utf-8') as f:
		for item in lines_to_write:
			f.write(item)
	with open(linking_filename, 'wb') as f:
		for key, vals in event_linking_dict.items():
			line_to_write = key + '\t'
			for i, val in enumerate(vals):
				if i == (len(vals) - 1):
					line_to_write += val
				else:
					line_to_write += (val + ' ')
			f.write(line_to_write + '\n')
	with codecs.open(logfile , 'a', encoding='utf-8') as f:
		f.write(write_filename + ', ' + str(len(lines_to_write)) + '\n')



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




def parallel_call_to_extract_from_docs_and_save_with_args(docs, filenames, save_dir, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, trigger_classifier_filename, classifier_dict_filename, realis_classifier_filename, raw_text_directory, crf_file):
	empty_trigger_labels = map(lambda x: [], xrange(len(docs)))
	docs_divided_among_cores, filenames_divided_among_cores, w2v_model_dicts_divided_among_cores = create_parallel_infrastructure_eal(docs, filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, crf_file)	
	p = Pool(num_cores)
	p.map(parallel_extract_from_docs_and_save_with_args, ((docs, filenames, w2v_model_dicts, save_dir, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, trigger_classifier_filename, classifier_dict_filename, realis_classifier_filename, raw_text_directory, crf_file) for docs, filenames, w2v_model_dicts in zip(docs_divided_among_cores, filenames_divided_among_cores, w2v_model_dicts_divided_among_cores)))
	#map(parallel_extract_from_docs_and_save_with_args, ((docs, filenames, w2v_model_dicts, save_dir, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, trigger_classifier_filename, classifier_dict_filename, realis_classifier_filename, raw_text_directory) for docs, filenames, w2v_model_dicts in zip(docs_divided_among_cores, filenames_divided_among_cores, w2v_model_dicts_divided_among_cores)))
	p.terminate()

#p = []
def set_pool(num_cores):
	global p
	p = Pool(num_cores)

use_blank_w2v = False