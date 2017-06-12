import Queue
import nom_utils
import operator
import nltk
from nltk.corpus import wordnet as wn
from joblib import Parallel, delayed
import multiprocessing
import ast
import csv
import pickle
import numpy as np
from math import sqrt

# not sure about HEADLINE and SUBJECT
tags_to_eliminate = ['<DOCID', '<DOCTYPE', '<DATETIME', '<HEADLINE', '<SPEAKER', '<ENDTIME', '<POSTDATE', '<POSTER', '<QUOTE', '<SUBJECT']

def replace_tags_with_spaces(text):
	new_text = ""
	open_tag = False
	for i, character in enumerate(text):
		if character == '<':
			open_tag = True
			new_text += ' '
		elif character == '>':
			open_tag = False
			new_text += ' '
		else:
			if open_tag:
				new_text += ' '
			else:
				new_text += character
	if len(new_text) != len(text):
		raise ValueError('Replacing SGML with spaces yielded different text length')
	return new_text

def replace_tags_and_metadata_with_spaces(text):
	new_text = ""
	open_tag = False
	current_word = ""
	for i, character in enumerate(text):
		if character == '<':
			open_tag = True
			new_text += '#'
			current_word = "<"
		elif character == '>':
			open_tag = False			
			for bad_tag in tags_to_eliminate:
				if bad_tag in current_word:
					if bad_tag != "<QUOTE":
						open_tag = True
					current_word = ""
			new_text += '#'
		else:
			if open_tag:
				new_text += '#'
				current_word += character
			else:
				new_text += character
	if len(new_text) != len(text):
		raise ValueError('Replacing SGML with spaces yielded different text length')
	return new_text

def create_string_from_tokens(tokens):
	s = ""
	for token in tokens:
		s += token['word'] + " "
	return s

def get_token(index, tokens):
	return tokens[index - 1]

def get_tokens_from_offsets(offset_range, tokens, adjust_offset_range = False):
	found_tokens = []
	begin = offset_range[0]
	end = offset_range[1]
	total_offset_adjust = 0
	if adjust_offset_range:
		for adjust in doc_offset_adjusts:
			if adjust[0] < end:
				total_offset_adjust += adjust[1]
		begin += total_offset_adjust
		end += total_offset_adjust			
	for token in tokens:
		#if token['characterOffsetBegin'] >= begin and token['characterOffsetEnd'] <= end:
		if token['characterOffsetBegin'] < end and token['characterOffsetEnd'] > begin:
			found_tokens.append(token)
	return found_tokens

def is_governor(token, token_indices, dependencies):
	for dep in dependencies:
		if dep['governor'] == token['index'] and dep['dependent'] in token_indices:
			return True
	return False

def is_not_governed(token, token_indices, dependencies):
	for dep in dependencies:
		if dep['dependent'] == token['index'] and dep['governor'] in token_indices:
			return False
	return True


# get linguistic head of a range of tokens
def get_head(phrase_tokens, tokens, dependencies):
	#phrase_tokens = get_tokens_from_offsets(offset_range, tokens)
	#phrase_tokens = offset_range
	if len(phrase_tokens) == 1:
		return phrase_tokens[0]
	elif len(phrase_tokens) == 0:
		return False
	phrase_token_indices = map(lambda x: x['index'], phrase_tokens)
	temp_head = phrase_tokens[0]
	checked_tokens = []
	for token in phrase_tokens[0:]:
		if is_governor(token, phrase_token_indices, dependencies) and is_not_governed(token, phrase_token_indices, dependencies):
			return token
	#print map(lambda x: x['word'], phrase_tokens), create_string_from_tokens(tokens)
	max_token_index = 0
	max_token_list_index = 0
	for i, token in enumerate(phrase_tokens):
		if token['index'] > max_token_index:
			max_token_index = token['index']
			max_token_list_index = i
	return phrase_tokens[max_token_list_index]


def find_shortest_path_to_token(token, dependencies, tokens, dep_type, pos_types, direction, arg_token, reduce_deps):
	#dep_mapping = get_dep_mapping()
	# if len(tokens) > 100:
	# 	return False
	visited = []
	node_q = Queue.Queue()		
	path_q = Queue.Queue()
	#token = get_token(this_dep['dependent'], tokens)
	#node_q.put((this_dep['governor'], this_dep['dependent']))
	node_q.put(token)
	#path_q.put(this_dep['dep'])
	path_q.put('token')
	#visited.append((this_dep['governor'], this_dep['dependent']))
	visited.append(token)
	#if direction == 'up':
	visit_count = 0
	while not node_q.empty():
		visit_count += 1
		current_token = node_q.get()
		current_path = path_q.get()
		current_index = current_token['index']
		#if current_token['pos'] in pos_types:
		if current_token == arg_token:
			return current_path
		if visit_count > 250 and len(current_path.split(' ')) > 10:
			#print 'breaking from find shortest path'
			return current_path
		elif visit_count > 350:
			#print 'breaking from find shortest path (length)'
			return current_path
		for node in dependencies:
			if node['governor'] == current_index:
				dep_token = get_token(node['dependent'], tokens)
				if dep_token not in visited:
					visited.append(dep_token)
					node_q.put(dep_token)
					if reduce_deps:
						if ":" in node['dep']:
							node['dep'] = node['dep'].split(":")[0]
						path_q.put(current_path + " down " + dep_mapping[node['dep']])
					else:
						path_q.put(current_path + " down " + node['dep'])
			elif node['dependent'] == current_index:
				gov_token = get_token(node['governor'], tokens)
				if gov_token not in visited:
					visited.append(gov_token)
					node_q.put(gov_token)
					if reduce_deps:
						if ":" in node['dep']:
							node['dep'] = node['dep'].split(":")[0]						
						path_q.put(current_path + " up " + dep_mapping[node['dep']])
					else:
						path_q.put(current_path + " up " + node['dep'])
	#print token['word'], arg_token['word'], create_string_from_tokens(tokens)
	return False	


def get_arg_paths(trigger, token, sentence, reduce_deps, dependency_type):
	arg_paths = []
	arg_texts = []
	for arg_index, arg in enumerate(trigger['args']):
		arg_tokens = get_tokens_from_offsets(arg, sentence['tokens'])
		if arg_tokens != []:
			#arg_head = get_head(arg_tokens, sentence['tokens'], sentence['basic-dependencies'])
			arg_head = get_head(arg_tokens, sentence['tokens'], sentence[dependency_type])
			arg_path = find_shortest_path_to_token(token, sentence[dependency_type], sentence['tokens'], None, ['ROOT', 'root'], 'both', arg_head, reduce_deps)
			arg_paths.append(arg_path)
			arg_texts.append(trigger['arg_texts'][arg_index])
	return arg_paths, arg_texts

def get_arg_paths_plus_wn_types(trigger, token, sentence, reduce_deps, dependency_type, add_word=True):
	arg_paths = []
	arg_texts = []
	for arg_index, arg in enumerate(trigger['args']):
		arg_tokens = get_tokens_from_offsets(arg, sentence['tokens'])
		for arg_head in arg_tokens:
			wn_type = "wn"				
			arg_path = find_shortest_path_to_token(token, sentence[dependency_type], sentence['tokens'], None, ['ROOT', 'root'], 'both', arg_head, reduce_deps)
			if arg_path:
				if add_word:
					arg_paths.append(arg_path + " wn " + wn_type + ' word ' + token['lemma'])
				else:
					arg_paths.append(arg_path + " wn " + wn_type + ' word ' + 'word')
				arg_texts.append(trigger['arg_texts'][arg_index])
	return arg_paths, arg_texts	

def get_path_plus_wn_types(trigger, token, sentence, reduce_deps, dependency_type, add_word=True):
	arg_paths = []
	print "trigger: ", trigger
	print 'token', token
	#arg_texts = []
	arg_path = find_shortest_path_to_token(token, sentence[dependency_type], sentence['tokens'], None, ['ROOT', 'root'], 'both', trigger, reduce_deps)
	print arg_path
	#for arg_index, arg in enumerate(trigger['args']):
	#	arg_tokens = get_tokens_from_offsets(arg, sentence['tokens'])
	#	if arg_tokens != []:
	#		#arg_head = get_head(arg_tokens, sentence['tokens'], sentence['basic-dependencies'])
	#		arg_head = get_head(arg_tokens, sentence['tokens'], sentence[dependency_type])
	#		if arg_head:
				#wn_types = get_hypernym_list(arg_head['lemma'])
	arg_head = token
	wn_type = get_wn_type(arg_head['lemma'])
	#		else:
	#			wn_type = ['NONE']
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
	if token['ner'] != 'O':
		wn_type = token['ner']				
	#		arg_path = find_shortest_path_to_token(token, sentence[dependency_type], sentence['tokens'], None, ['ROOT', 'root'], 'both', arg_head, reduce_deps)
	if arg_path:
		if add_word:
			arg_paths.append(arg_path + " wn " + wn_type + ' word ' + token['lemma'])
		else:
			arg_paths.append(arg_path + " wn " + wn_type + ' word ' + 'word')
	#arg_texts.append(trigger['arg_texts'][arg_index])
	return arg_paths	

def is_arg(token, triggers, sentence):
	for trigger in triggers:
		for arg in trigger['args']:
			arg_tokens = get_tokens_from_offsets(arg, sentence['tokens'])
			if token in arg_tokens:
				return True
	return False

def get_non_arg_noun_paths(token, sentence, triggers, reduce_deps, dependency_type):
	tokens = sentence['tokens']
	dependencies = sentence[dependency_type]
	arg_list = []
	arg_string_list = []
	for comp_token in tokens:
		if comp_token == token:
			continue
		if comp_token['pos'] not in noun_types:
			continue
		#if is_arg(comp_token, triggers, sentence):
		#	continue			
		wn_type = get_wn_type(comp_token['lemma'])
		if 'person' in wn_type:
			wn_type = "PERSON"
		elif 'location' in wn_type:
			wn_type = 'LOCATION'
		elif 'time' in wn_type:
			wn_type = 'TIME'
		elif 'group' in wn_type:
			wn_type = "ORGANIZATION"
		else:
			wn_type = 'noun'
		if token['ner'] != 'O':
			wn_type = token['ner']				
		path = find_shortest_path_to_token(token, dependencies, tokens, None, [], 'both', comp_token, reduce_deps)
		if path:
			arg_list.append(path + " wn " + wn_type + ' word ' + 'word')
			arg_string_list.append(comp_token['word'])
	return arg_list, arg_string_list

def path_list_to_dict(path_list):
	path_dict = {}
	for path in path_list:
		if path in path_dict:
			path_dict[path] += 1
		else:
			path_dict[path] = 1	
	return path_dict

def path_list_total(path_dict):
	total_path = 0
	for key, val in path_dict.items():
		total_path += val
	return total_path

def sort_and_print_paths(path_dict, path_total, num_to_print = 0):
	sorted_x = sorted(path_dict.items(), key=operator.itemgetter(1), reverse=True)
	if num_to_print == 0:
		for x in sorted_x:
			print x, path_dict[x[0]], float(path_dict[x[0]]) / path_total 		
	else:
		for x in sorted_x[:num_to_print]:
			print x, path_dict[x[0]], float(path_dict[x[0]]) / path_total		



def get_dep_mapping():
	dep_mapping = {}
	dep_mapping['nsubj'] = "subj"
	dep_mapping['nsubjpass'] = "subj"
	dep_mapping['dobj'] = "dobj"
	dep_mapping['iobj'] = "iobj"
	dep_mapping['csubj'] = "subj"
	dep_mapping['csubjpass'] = "subj"
	dep_mapping['ccomp'] = "comp"
	dep_mapping['xcomp'] = "comp"
	dep_mapping['nummod'] = "nummod"
	dep_mapping['appos'] = "mod"
	dep_mapping['nmod'] = "mod"
	dep_mapping['acl'] = "mod"
	dep_mapping['amod'] = "mod"
	dep_mapping['det'] = "det"
	dep_mapping['neg'] = "neg"
	dep_mapping['case'] = "case"
	dep_mapping['advcl'] = "mod"
	dep_mapping['advmod'] = "mod"
	dep_mapping['compound'] = "compound"
	dep_mapping['name'] = "compound"
	dep_mapping['mwe'] = "compound"
	dep_mapping['foreign'] = "foreign"
	dep_mapping['goeswith'] = "goeswith"
	dep_mapping['list'] = "list"
	dep_mapping['dislocated'] = "dislocated"
	dep_mapping['parataxis'] = "parataxis"
	dep_mapping['remnant'] = "other"
	dep_mapping['reparandum'] = "other"
	dep_mapping['vocative'] = "dobj"
	dep_mapping['discourse'] = "foreign"
	dep_mapping['expl'] = "expl"
	dep_mapping['aux'] = "aux"
	dep_mapping['auxpass'] = "aux"
	dep_mapping['cop'] = "cop"
	dep_mapping['mark'] = "mark"
	dep_mapping['punct'] = "punct"
	dep_mapping['conj'] = "conj"
	dep_mapping['cc'] = "conj"
	dep_mapping['root'] = "ROOT"
	dep_mapping['ROOT'] = "ROOT"
	dep_mapping['dep'] = "dep"	
	dep_mapping['NONE'] = "NONE"
	return dep_mapping

def get_dep_mapping2():
	dep_mapping = {}
	dep_mapping['nsubj'] = "subj"
	dep_mapping['nsubjpass'] = "subj"
	dep_mapping['dobj'] = "dobj"
	dep_mapping['iobj'] = "iobj"
	dep_mapping['csubj'] = "subj"
	dep_mapping['csubjpass'] = "subj"
	dep_mapping['ccomp'] = "comp"
	dep_mapping['xcomp'] = "comp"
	dep_mapping['nummod'] = "nummod"
	dep_mapping['appos'] = "mod"
	dep_mapping['nmod'] = "mod"
	dep_mapping['acl'] = "mod"
	dep_mapping['amod'] = "mod"
	dep_mapping['det'] = "det"
	dep_mapping['neg'] = "neg"
	dep_mapping['case'] = "case"
	dep_mapping['advcl'] = "mod"
	dep_mapping['advmod'] = "mod"
	dep_mapping['compound'] = "compound"
	dep_mapping['name'] = "compound"
	dep_mapping['mwe'] = "compound"
	dep_mapping['foreign'] = "foreign"
	dep_mapping['goeswith'] = "goeswith"
	dep_mapping['list'] = "list"
	dep_mapping['dislocated'] = "dislocated"
	dep_mapping['parataxis'] = "parataxis"
	dep_mapping['remnant'] = "other"
	dep_mapping['reparandum'] = "other"
	dep_mapping['vocative'] = "dobj"
	dep_mapping['discourse'] = "foreign"
	dep_mapping['expl'] = "expl"
	dep_mapping['aux'] = "aux"
	dep_mapping['auxpass'] = "aux"
	dep_mapping['cop'] = "cop"
	dep_mapping['mark'] = "mark"
	dep_mapping['punct'] = "punct"
	dep_mapping['conj'] = "conj"
	dep_mapping['cc'] = "conj"
	dep_mapping['root'] = "ROOT"
	dep_mapping['ROOT'] = "ROOT"
	dep_mapping['dep'] = "dep"	
	return dep_mapping

dep_mapping = get_dep_mapping()
noun_types = nom_utils.get_noun_types()

def get_all_pos_tags():
	return ['NONE', 'CC', 'CD', 'DT', 'EX', 'FW', 'IN', 'JJ', 'JJR', 'JJS', 'LS', 'MD', 'NN', 'NNS', 'NNP', 'NNPS', 'PDT', 'POS', 'PRP', 'PRP$', 'RB', 'RBR', 'RBS', 'RP', 'SYM', 'TO', 'UH', 'VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ', 'WDT', 'WP', 'WP$', 'WRB']

def get_event_word_counts(doc, event_word_dict, filename):
	event_word_counts = {}
	token_count = 0
	word_list = {}
	for event in canonical_event_list:
		event_word_counts[event] = 0
		word_list[event] = []
	#for event_list in event_word_dict.values():
	#	for event in event_list:
	#		event_word_counts[event] = 0
	for sentence in doc['sentences']:
		tokens = sentence['tokens']
		for token in tokens:
			token_count += 1
			word = token['word']
			lemma = token['lemma']
			if word in event_word_dict:
				for event in event_word_dict[word]:
					event_word_counts[event] += 1
					word_list[event].append(word)
			elif lemma in event_word_dict:
				for event in event_word_dict[lemma]:
					event_word_counts[event] += 1
					word_list[event].append(lemma)
	if token_count > 0:									
		for event, count in event_word_counts.items():
			event_word_counts[event] = float(count) / token_count
	else:
		print "No tokens found in: " + filename
	return event_word_counts, word_list

def get_all_events_from_triggers(triggers):
	all_events = set()
	for trigger in triggers:
		event = trigger['ere_type'] + "." + trigger['subtype']
		all_events.add(event)
	return all_events

canonical_event_list = []
canonical_event_list.append('PersonnelEvent.Elect')
canonical_event_list.append('JusticeEvent.Pardon')
canonical_event_list.append('JusticeEvent.Sentencing')
canonical_event_list.append('Movement.Transport.TransportArtifact')
canonical_event_list.append('Movement.Transport.TransportPerson')
canonical_event_list.append('JusticeEvent.TrialHearing')
canonical_event_list.append('JusticeEvent.ChargeIndict')
canonical_event_list.append('BusinessEvent.EndOrganization')
canonical_event_list.append('JusticeEvent.Fine')
canonical_event_list.append('PersonnelEvent.EndPosition')
canonical_event_list.append('JusticeEvent.Sue')
canonical_event_list.append('JusticeEvent.Acquit')
canonical_event_list.append('BusinessEvent.DeclareBankruptcy')
canonical_event_list.append('JusticeEvent.Extradite')
canonical_event_list.append('LifeEvent.Divorce')
canonical_event_list.append('JusticeEvent.ArrestJail')
canonical_event_list.append('Conflict.Attack')
canonical_event_list.append('JusticeEvent.Convict')
canonical_event_list.append('JusticeEvent.Appeal')
canonical_event_list.append('Manufacture.Artifact')
canonical_event_list.append('Transaction.TransferOwnership')
canonical_event_list.append('JusticeEvent.Execute')
canonical_event_list.append('Contact.Broadcast')
canonical_event_list.append('LifeEvent.BeBorn')
canonical_event_list.append('PersonnelEvent.StartPosition')
canonical_event_list.append('LifeEvent.Die')
canonical_event_list.append('LifeEvent.Marry')
canonical_event_list.append('LifeEvent.Injure')
canonical_event_list.append('BusinessEvent.MergeOrg')
canonical_event_list.append('PersonnelEvent.Nominate')
canonical_event_list.append('Conflict.Demonstrate')
canonical_event_list.append('JusticeEvent.ReleaseParole')
canonical_event_list.append('Contact.Meet')
#canonical_event_list.append('Contact.Correspondence')


def get_canonical_event_list():
	return canonical_event_list

deft_ace_mapping = {}

deft_ace_mapping['PersonnelEvent.Elect'] = "Personnel.Elect"
deft_ace_mapping['JusticeEvent.ArrestJail'] = "Justice.Arrest-Jail"
#deft_ace_mapping['Contact.Broadcast'] = "Contact.Phone-Write" ## HACK NEED REAL EVENT
deft_ace_mapping['BusinessEvent.EndOrganization'] = "Business.End-Org"
deft_ace_mapping['JusticeEvent.Sue'] = "Justice.Sue"
deft_ace_mapping['JusticeEvent.Appeal'] = "Justice.Appeal"
deft_ace_mapping['LifeEvent.BeBorn'] = "Life.Be-Born"
deft_ace_mapping['LifeEvent.Die'] = "Life.Die"
deft_ace_mapping['JusticeEvent.Convict'] = "Justice.Convict"
deft_ace_mapping['Transaction.TransferMoney'] = "Transaction.Transfer-Money" ## HACK NEED REAL EVENT
deft_ace_mapping['PersonnelEvent.StartPosition'] = "Personnel.Start-Position"
deft_ace_mapping['JusticeEvent.TrialHearing'] = "Justice.Trial-Hearing"
deft_ace_mapping['Movement.Transport.TransportPerson'] = "Movement.Transport"
deft_ace_mapping['LifeEvent.Marry'] = "Life.Marry"
deft_ace_mapping['Contact.Meet'] = "Contact.Meet"
deft_ace_mapping['PersonnelEvent.EndPosition'] = "Personnel.End-Position"
deft_ace_mapping['JusticeEvent.Sentencing'] = "Justice.Sentence"
deft_ace_mapping['JusticeEvent.Execute'] = "Justice.Execute"
deft_ace_mapping['Transaction.TransferOwnership'] = "Transaction.Transfer-Ownership"
deft_ace_mapping['PersonnelEvent.Nominate'] = "Personnel.Nominate"
deft_ace_mapping['JusticeEvent.Extradite'] = "Justice.Extradite"
deft_ace_mapping['HACK1'] = "Business.Start-Org"  ## HACK NEED REAL EVENT
deft_ace_mapping['LifeEvent.Divorce'] = "Life.Divorce"
deft_ace_mapping['JusticeEvent.Fine'] = "Justice.Fine"
deft_ace_mapping['JusticeEvent.Acquit'] = "Justice.Acquit"
deft_ace_mapping['BusinessEvent.MergeOrg'] = "Business.Merge-Org"
deft_ace_mapping['LifeEvent.Injure'] = "Life.Injure"
deft_ace_mapping['Conflict.Attack'] = "Conflict.Attack"
deft_ace_mapping['BusinessEvent.DeclareBankruptcy'] = "Business.Declare-Bankruptcy"
deft_ace_mapping['JusticeEvent.ChargeIndict'] = "Justice.Charge-Indict"
deft_ace_mapping['JusticeEvent.Pardon'] = "Justice.Pardon"
deft_ace_mapping['JusticeEvent.ReleaseParole'] = "Justice.Release-Parole"
deft_ace_mapping['Conflict.Demonstrate'] = "Conflict.Demonstrate"
deft_ace_mapping['Contact.Correspondence'] = "Contact.Phone-Write"


deft_eal_mapping = {}
deft_eal_mapping['PersonnelEvent.Elect'] = "Personnel.Elect"
deft_eal_mapping['JusticeEvent.ArrestJail'] = "Justice.Arrest-Jail"
deft_eal_mapping['Contact.Broadcast'] = "Contact.Broadcast" ## HACK NEED REAL EVENT
deft_eal_mapping['BusinessEvent.EndOrganization'] = "NONE"
deft_eal_mapping['JusticeEvent.Sue'] = "NONE"
deft_eal_mapping['JusticeEvent.Appeal'] = "NONE"
deft_eal_mapping['LifeEvent.BeBorn'] = "NONE"
deft_eal_mapping['LifeEvent.Die'] = "Life.Die"
deft_eal_mapping['JusticeEvent.Convict'] = "NONE"
deft_eal_mapping['Transaction.TransferMoney'] = "Transaction.Transfer-Money" ## HACK NEED REAL EVENT
deft_eal_mapping['PersonnelEvent.StartPosition'] = "Personnel.Start-Position"
deft_eal_mapping['JusticeEvent.TrialHearing'] = "NONE"
deft_eal_mapping['Movement.Transport.TransportPerson'] = "Movement.Transport-Person"
deft_eal_mapping['LifeEvent.Marry'] = "NONE"
deft_eal_mapping['Contact.Meet'] = "Contact.Meet"
deft_eal_mapping['PersonnelEvent.EndPosition'] = "Personnel.End-Position"
deft_eal_mapping['JusticeEvent.Sentencing'] = "NONE"
deft_eal_mapping['JusticeEvent.Execute'] = "NONE"
deft_eal_mapping['Transaction.TransferOwnership'] = "Transaction.Transfer-Ownership"
deft_eal_mapping['PersonnelEvent.Nominate'] = "NONE"
deft_eal_mapping['JusticeEvent.Extradite'] = "NONE"
deft_eal_mapping['HACK1'] = "NONE"  ## HACK NEED REAL EVENT was business.start-org
deft_eal_mapping['LifeEvent.Divorce'] = "NONE"
deft_eal_mapping['JusticeEvent.Fine'] = "NONE"
deft_eal_mapping['JusticeEvent.Acquit'] = "NONE"
deft_eal_mapping['BusinessEvent.MergeOrg'] = "NONE"
deft_eal_mapping['LifeEvent.Injure'] = "Life.Injure"
deft_eal_mapping['Conflict.Attack'] = "Conflict.Attack"
deft_eal_mapping['BusinessEvent.DeclareBankruptcy'] = "NONE"
deft_eal_mapping['JusticeEvent.ChargeIndict'] = "NONE"
deft_eal_mapping['JusticeEvent.Pardon'] = "NONE"
deft_eal_mapping['JusticeEvent.ReleaseParole'] = "NONE"
deft_eal_mapping['Conflict.Demonstrate'] = "Conflict.Demonstrate"
deft_eal_mapping['Contact.Correspondence'] = "Contact.Correspondence"
deft_eal_mapping['Manufacture.Artifact'] = "Manufacture.Artifact"
deft_eal_mapping['Movement.Transport.TransportArtifact'] = "Movement.Transport-Artifact"


rere_lower_to_upper_mapping = {}
rere_lower_to_upper_mapping['personnel.elect'] = "Personnel.Elect"
rere_lower_to_upper_mapping['justice.arrestjail'] = "Justice.Arrest-Jail"
rere_lower_to_upper_mapping['contact.broadcast'] = "Contact.Broadcast" ## HACK NEED REAL EVENT
rere_lower_to_upper_mapping['business.endorg'] = "NONE"
rere_lower_to_upper_mapping['justice.sue'] = "NONE"
rere_lower_to_upper_mapping['justice.appeal'] = "NONE"
rere_lower_to_upper_mapping['life.beborn'] = "NONE"
rere_lower_to_upper_mapping['life.die'] = "Life.Die"
rere_lower_to_upper_mapping['justice.convict'] = "NONE"
rere_lower_to_upper_mapping['transaction.transfermoney'] = "Transaction.Transfer-Money" ## HACK NEED REAL EVENT
rere_lower_to_upper_mapping['personnel.startposition'] = "Personnel.Start-Position"
rere_lower_to_upper_mapping['justice.trialhearing'] = "NONE"
rere_lower_to_upper_mapping['movement.transportperson'] = "Movement.Transport-Person"
rere_lower_to_upper_mapping['life.marry'] = "NONE"
rere_lower_to_upper_mapping['contact.meet'] = "Contact.Meet"
rere_lower_to_upper_mapping['personnel.endposition'] = "Personnel.End-Position"
rere_lower_to_upper_mapping['justice.sentencing'] = "NONE"
rere_lower_to_upper_mapping['justice.sentence'] = "NONE"
rere_lower_to_upper_mapping['justice.execute'] = "NONE"
rere_lower_to_upper_mapping['transaction.transferownership'] = "Transaction.Transfer-Ownership"
rere_lower_to_upper_mapping['personnel.nominate'] = "NONE"
rere_lower_to_upper_mapping['justice.extradite'] = "NONE"
rere_lower_to_upper_mapping['HACK1'] = "NONE"  ## HACK NEED REAL EVENT was business.start-org
rere_lower_to_upper_mapping['life.divorce'] = "NONE"
rere_lower_to_upper_mapping['justice.fine'] = "NONE"
rere_lower_to_upper_mapping['justice.acquit'] = "NONE"
rere_lower_to_upper_mapping['business.mergeorg'] = "NONE"
rere_lower_to_upper_mapping['life.injure'] = "Life.Injure"
rere_lower_to_upper_mapping['conflict.attack'] = "Conflict.Attack"
rere_lower_to_upper_mapping['business.declarebankruptcy'] = "NONE"
rere_lower_to_upper_mapping['justice.chargeindict'] = "NONE"
rere_lower_to_upper_mapping['justice.pardon'] = "NONE"
rere_lower_to_upper_mapping['justice.releaseparole'] = "NONE"
rere_lower_to_upper_mapping['conflict.demonstrate'] = "Conflict.Demonstrate"
rere_lower_to_upper_mapping['contact.correspondence'] = "Contact.Correspondence"
rere_lower_to_upper_mapping['contact.communicate'] = "Contact.Correspondence"
rere_lower_to_upper_mapping['manufacture.artifact'] = "Manufacture.Artifact"
rere_lower_to_upper_mapping['movement.transportartifact'] = "Movement.Transport-Artifact"
rere_lower_to_upper_mapping['business.startorg'] = "NONE"
rere_lower_to_upper_mapping['justice.tryholdhearing'] = "NONE"
rere_lower_to_upper_mapping['contact.phone-write'] = "Contact.Correspondence"
rere_lower_to_upper_mapping['transaction.transaction'] = "Transaction.Transaction"
rere_lower_to_upper_mapping['contact.contact'] = "Contact.Contact"

def get_deft_ace_mapping():
	return deft_ace_mapping

ace_deft_mapping = {}
for key, val in deft_ace_mapping.items():
	ace_deft_mapping[val] = key


ace_eal_mapping = {}
for ace, deft in ace_deft_mapping.items():
	eal = deft_eal_mapping[deft]
	ace_eal_mapping[ace] = eal

def get_ace_deft_mapping():
	return ace_deft_mapping


def sort_dict_by_key_return_val(source_dict):
	sorted_keys = sorted(source_dict.keys())
	return map(lambda x: source_dict[x], sorted_keys)

def trigger_matches_token(token, trigger):
	trigger_offset_begin = trigger['offset']
	trigger_offset_end = trigger_offset_begin + trigger['length']
	#charseq_offset_begin = trigger['full_charseq_offset_start']
	#charseq_offset_end = charseq_offset_begin + trigger['full_charseq_length']							
	if trigger_offset_begin < token['characterOffsetEnd'] and trigger_offset_end > token['characterOffsetBegin']:
		if token['word'].lower() not in trigger['trigger'].lower():
			print "ERROR:", token['word'], 'not found in', trigger['trigger']
			#return False
		return True
	return False

def extent_matches_token(token, trigger):
	trigger_offset_begin = trigger['offset']
	trigger_offset_end = trigger_offset_begin + trigger['length']
	charseq_offset_begin = trigger['full_charseq_offset_start']
	charseq_offset_end = charseq_offset_begin + trigger['full_charseq_length']							
	if charseq_offset_begin < token['characterOffsetEnd'] and charseq_offset_end > token['characterOffsetBegin']:
		if token['word'].lower() not in trigger['full_charseq'].lower():
			print "ERROR:", token['word'], 'not found in', trigger['full_charseq']
			#return False
		return True
	return False		

def get_triggers_for_token(token, trigger_labels, extent=False):
	found_trigger = False
	trigger_match = "NONE"
	trigger_event = "NONE"
	matching_triggers = []
	for trigger in trigger_labels:
		if extent:
			if extent_matches_token(token, trigger):
				matching_triggers.append(trigger)
		else:
			if trigger_matches_token(token, trigger):
				matching_triggers.append(trigger)
	return matching_triggers

def entity_matches_token(token, entity):
	begin, end = entity['entity_offsets']
	if begin < token['characterOffsetEnd'] and end > token['characterOffsetBegin']:
		if token['word'].lower() not in entity['entity_text'].lower():
			print "ERROR:", token['word'], 'not found in', entity['entity_text']
			#return False
		return True
	return False	

def get_entities_for_token(token, entity_labels):
	found_trigger = False
	trigger_match = "NONE"
	trigger_event = "NONE"
	matching_entities = []
	for entity in entity_labels:
		if entity_matches_token(token, entity):
			matching_entities.append(entity)
	return matching_entities	

def gather_arg_deps(docs, trigger_info, indices, reduce_deps, dependency_type, min_occurrences):
	arg_paths = {}
	for i in indices:
		doc = docs[i]
		doc_triggers = trigger_info[i]
		for sentence in doc['sentences']:
			for token in sentence['tokens']:
				if token['pos'] not in nom_utils.get_noun_types():
					continue
				# Skip words < three letters (like 'a.' or '1.')
				if len(token['word']) <= 3:
					continue
				triggers = get_triggers_for_token(token, doc_triggers)
				for trigger in triggers:
					this_arg_paths, this_arg_strings = get_arg_paths_plus_wn_types(trigger, token, sentence, reduce_deps, dependency_type)
					for path in this_arg_paths:
						if path in arg_paths:
							arg_paths[path] += 1
						else:
							arg_paths[path] = 1
	arg_paths_over_threshold = []
	for path, count in arg_paths.items():
		if count > min_occurrences:
			arg_paths_over_threshold.append(path)
	return arg_paths_over_threshold

def gather_noun_deps(docs, trigger_info, indices, reduce_deps, dependency_type, parallelize_compute, min_occurrences):
	arg_paths = {}
	for i in indices:
		doc = docs[i]
		doc_triggers = trigger_info[i]
		if not isinstance(doc, dict):
			return []		
		for sentence in doc['sentences']:
			for token in sentence['tokens']:
				if token['pos'] not in nom_utils.get_noun_types():
					continue
				# Skip words < three letters (like 'a.' or '1.')
				if len(token['word']) <= 3:
					continue
				triggers = get_triggers_for_token(token, doc_triggers)
				#for trigger in triggers:
				#	this_arg_paths, this_arg_strings = get_arg_paths_plus_wn_types(trigger, token, sentence, reduce_deps, dependency_type)
				this_arg_paths, this_arg_strings = get_non_arg_noun_paths(token, sentence, doc_triggers, reduce_deps, dependency_type)
				for path in this_arg_paths:
					if path in arg_paths:
						arg_paths[path] += 1
					else:
						arg_paths[path] = 1
	arg_paths_over_threshold = []
	for path, count in arg_paths.items():
		if count > min_occurrences:
			arg_paths_over_threshold.append(path)
	return arg_paths_over_threshold

def gather_person_deps(docs, trigger_info, indices, reduce_deps, dependency_type, parallelize_compute, min_occurrences):
	arg_paths = {}
	for i in indices:
		doc = docs[i]
		doc_triggers = trigger_info[i]
		if not isinstance(doc, dict):
			return []		
		for sentence in doc['sentences']:
			for token in sentence['tokens']:
				if token['pos'] not in nom_utils.get_noun_types():
					continue
				# Skip words < three letters (like 'a.' or '1.')
				if len(token['word']) <= 3:
					continue
				triggers = get_triggers_for_token(token, doc_triggers)
				#for trigger in triggers:
				#	this_arg_paths, this_arg_strings = get_arg_paths_plus_wn_types(trigger, token, sentence, reduce_deps, dependency_type)
				this_arg_paths = get_trigger_and_nontrigger_person_paths(token, sentence, doc_triggers, dependency_type, reduce_deps)
				for path in this_arg_paths:
					if path in arg_paths:
						arg_paths[path] += 1
					else:
						arg_paths[path] = 1
	arg_paths_over_threshold = []
	for path, count in arg_paths.items():
		if count > min_occurrences:
			arg_paths_over_threshold.append(path)
	return arg_paths_over_threshold



def get_doc_noun_deps(doc, doc_triggers, reduce_deps, dependency_type, nouns_only):
	if not isinstance(doc, dict):
		return []		
	path_list = []		
	for sentence in doc['sentences']:
		for token in sentence['tokens']:
			if nouns_only and token['pos'] not in nom_utils.get_noun_types():
				continue
			# Skip words < three letters (like 'a.' or '1.')
			if len(token['word']) <= 3:
				continue
			triggers = get_triggers_for_token(token, doc_triggers)
			#for trigger in triggers:
			#	this_arg_paths, this_arg_strings = get_arg_paths_plus_wn_types(trigger, token, sentence, reduce_deps, dependency_type)
			this_arg_paths, this_arg_strings = get_non_arg_noun_paths(token, sentence, doc_triggers, reduce_deps, dependency_type)
			path_list += this_arg_paths
	return path_list	



def get_doc_trigger_deps(doc, doc_triggers, reduce_deps, dependency_type, nouns_only):
	# Used for arg extractor
	if not isinstance(doc, dict):
		return []		
	path_list = []		
	for sentence in doc['sentences']:
		for token in sentence['tokens']:
			if nouns_only and token['pos'] not in nom_utils.get_noun_types():
				continue
			# Skip words < three letters (like 'a.' or '1.')
			if len(token['word']) <= 3:
				continue
			triggers = get_triggers_for_token(token, doc_triggers)
			#if token in triggers:
			#	continue
			for trigger in triggers:
				#offset_range = [int(trigger['offset']), int(trigger['offset']) + int(trigger['length'])]
				#trigger_token = get_tokens_from_offsets(offset_range, sentence['tokens'])
				#trigger_token = token
				this_arg_paths, _ = get_arg_paths_plus_wn_types(trigger, token, sentence, reduce_deps, dependency_type, False)
				#for check_token in sentence['tokens']:
				#	this_arg_paths = get_path_plus_wn_types(trigger_token, check_token, sentence, reduce_deps, dependency_type, False)
				path_list += this_arg_paths
			#for trigger in triggers:
			#	this_arg_paths, this_arg_strings = get_arg_paths_plus_wn_types(trigger, token, sentence, reduce_deps, dependency_type)
			#this_arg_paths = get_trigger_and_nontrigger_person_paths(token, sentence, doc_triggers, dependency_type, reduce_deps)
	return path_list	



def get_doc_person_deps(doc, doc_triggers, reduce_deps, dependency_type, nouns_only):
	if not isinstance(doc, dict):
		return []		
	path_list = []		
	for sentence in doc['sentences']:
		for token in sentence['tokens']:
			if nouns_only and token['pos'] not in nom_utils.get_noun_types():
				continue
			# Skip words < three letters (like 'a.' or '1.')
			if len(token['word']) <= 3:
				continue
			triggers = get_triggers_for_token(token, doc_triggers)
			#for trigger in triggers:
			#	this_arg_paths, this_arg_strings = get_arg_paths_plus_wn_types(trigger, token, sentence, reduce_deps, dependency_type)
			this_arg_paths = get_trigger_and_nontrigger_person_paths(token, sentence, doc_triggers, dependency_type, reduce_deps)
			path_list += this_arg_paths
	return path_list	

def gather_deps_par(docs, trigger_info, indices, reduce_deps, dependency_type, parallelize_compute, nouns_only, topk, dep_function, num_cores):
	if not parallelize_compute:
		num_cores = 1
	arg_paths_dict = {}
	doc_arg_paths = Parallel(n_jobs = num_cores)(delayed(dep_function)(doc, triggers, reduce_deps, dependency_type, nouns_only) for doc, triggers in zip(docs, trigger_info))
	for arg_path_list in doc_arg_paths:
		for arg_path in arg_path_list:
			if arg_path in arg_paths_dict:
				arg_paths_dict[arg_path] += 1
			else:
				arg_paths_dict[arg_path] = 1
	sorted_dict = sorted(arg_paths_dict.items(), key=operator.itemgetter(1), reverse=True)[:topk]
	arg_paths_over_threshold = map(lambda x: x[0], sorted_dict)
	#arg_paths_over_threshold = []
	#for path, count in arg_paths_dict.items():
	#	if count > min_occurrences:
	#		arg_paths_over_threshold.append(path)
	return arg_paths_over_threshold


def gather_doc_word_bigrams(doc, use_lemmas):
	if not isinstance(doc, dict):
		return set()	
	bigrams = set()
	for sentence in doc['sentences']:
		prev_word = "ROOT"
		for token in sentence['tokens']:
			if use_lemmas:
				token_word = token['lemma']
			else:
				token_word = token['word']
			bigram = prev_word + " " + token_word
			bigrams.add(bigram)
			prev_word = token_word
	bigrams.add(prev_word + " " + "END")
	return bigrams


def gather_word_bigrams(docs, use_lemmas, parallelize_compute, num_cores):
	if not parallelize_compute:
		num_cores = 1
	all_bigram_set = set()
	bigram_doc_list = Parallel(n_jobs = num_cores)(delayed(gather_doc_word_bigrams)(doc, use_lemmas) for doc in docs)
	for bigram_set in bigram_doc_list:
		all_bigram_set = all_bigram_set.union(bigram_set)
	return all_bigram_set


def get_hypernym_list(word):
	# right now just using first synset?
	if len(wn.synsets(word)) == 0:
		return []
	word_synsets = [wn.synsets(word)[0]]
	hypernyms = set()
	remaining_words = Queue.Queue()
	for sense in word_synsets:
		remaining_words.put(sense)
	while not remaining_words.empty():
		current_word = remaining_words.get()
		hypernyms.add(current_word.name())
		for hyp in current_word.hypernyms():
			remaining_words.put(hyp)
	return hypernyms

def get_wn_type(word):
	#senses = wn.synsets(word)
	#if len(senses) == 0:
	#	return "NONE"
	#sense = senses[0]
	#return sense.lexname()
	synset = get_most_likely_synset(word)
	if synset:
		return synset.lexname
	return "NONE"

def get_wn_types(word):
	#senses = wn.synsets(word)
	#if len(senses) == 0:
	#	return "NONE"
	#sense = senses[0]
	#return sense.lexname()
	wn_types = []
	try:
		synsets = wn.synsets(word)
	except(UnicodeDecodeError):
		print "unicode error, the world is a nightmare, caused by ", word
		synsets = []
	if len(synsets) == 0:
		return ['NONE']
	for synset in synsets:
		wn_types.append(synset.lexname)
	return wn_types


def get_most_likely_synset(word):
	try:
		synsets = wn.synsets(word)
	except(UnicodeDecodeError):
		print "unicode error, the world is a nightmare, caused by ", word
		synsets = []
	if len(synsets) == 0:
		return False
	max_count = 0
	max_synset = 0
	num_to_check = min(50, len(synsets))
	for i, s in enumerate(synsets[:num_to_check]):
		freq = 0  
		for lemma in s.lemmas:
			freq+=lemma.count()
		if freq > max_count:
			max_count = freq
			max_synset = i
		#sense2freq[i] = freq
	return synsets[max_synset]


def get_all_person_tokens(tokens):
	person_tokens = []
	for token in tokens:
		if len(person_tokens) > 20:
			return person_tokens
		if token['ner'] == "PERSON":
			person_tokens.append(token)
		else:
			#wn_type = get_wn_type(token['lemma'])
			#types = [wn_type]
			types = get_wn_types(token['lemma'])
			for wn_type in types:
				if 'person' in wn_type and token not in person_tokens:
					person_tokens.append(token)
	return person_tokens

def get_trigger_and_nontrigger_person_paths(token, sentence, triggers, dependency_type, reduce_deps):
	person_tokens = get_all_person_tokens(sentence['tokens'])
	person_arg_paths = []
	person_non_arg_paths = []
	for person_token in person_tokens:
		if person_token == token:
			continue
		path = find_shortest_path_to_token(token, sentence[dependency_type], sentence['tokens'], None, [], 'both', person_token, reduce_deps)
		path = path + " wn " + 'person' + ' word ' + 'word'
		if is_arg(person_token, triggers, sentence):
			person_arg_paths.append(path)
		else:
			person_non_arg_paths.append(path)
	return person_arg_paths + person_non_arg_paths

def get_sentence_dict(s):
	if isinstance(s, dict):
		return s
	print "had to fix"
	s = filter(lambda x: x not in '\x00', s)
	d = ast.literal_eval(s)
	return d


def compute_precision_recall(predictions, true_labels, savefile = False, missed_triggers=0, event_list = False):
	if len(predictions) != len(true_labels):
		print "Lengths don't match"
		return
	tp = 0.
	fp = 0.
	tn = 0.
	fn = float(missed_triggers)
	for i in xrange(len(predictions)):
		pred = predictions[i]
		label = true_labels[i]
		if event_list:
			if pred not in event_list:
				pred = "NONE"
			if label not in event_list:
				label = "NONE"
		if label == "NONE":
			if pred == "NONE":
				tn += 1
			else:
				fp += 1
		else:
			if pred == label:
				tp += 1
			elif pred == "NONE":
				fn += 1
			else:
				fp += 1
				fn += 1
	print "TP: ", tp
	print "FP: ", fp
	print "TN: ", tn
	print "FN: ", fn
	if (tp + fp) > 0:				
		precision = tp / (tp + fp)
	else:
		precision = 0
	if (tp + fn) > 0:		
		recall = tp / (tp + fn)
	else:
		recall = 0
	if (precision + recall) > 0:
		f1 = 2 * precision * recall / (precision + recall)
	else:
		f1 = 0
	print "precision: ", precision
	print "recall: ", recall
	print "f1: ", f1
	if savefile:
		with open(savefile, 'wb') as f:
			writer = csv.writer(f)
			writer.writerow(['tp', tp])
			writer.writerow(['fp', fp])
			writer.writerow(['tn', tn])
			writer.writerow(['fn', fn])
			writer.writerow(['precision', precision])
			writer.writerow(['recall', recall])
			writer.writerow(['f1', f1])
	print "Done"

def compute_precision_recall_proba(predictions, true_labels, classifier, threshold=0.3, missed_triggers=0, event_list = False):
	if len(predictions) != len(true_labels):
		print "Lengths don't match"
		return
	tp = 0.
	fp = 0.
	tn = 0.
	fn = float(missed_triggers)
	for i in xrange(len(predictions)):
		pred = predictions[i]
		label = true_labels[i]	
		positive_classes = []
		for i, category in enumerate(classifier.classes_):
			if pred[i] > threshold:
				positive_classes.append(category)
		if len(positive_classes) == 0:
			positive_classes = ["NONE"]
		for category in positive_classes:
			if event_list:
				if category not in event_list:
					category = "NONE"
				if label not in event_list:
					label = "NONE"				
			if label == "NONE":
				if category == "NONE":
					tn += 1
				else:
					fp += 1
			else:
				if category == label:
					tp += 1
				elif category == "NONE":
					fn += 1
				else:
					fp += 1
					fn += 1
	print "TP: ", tp
	print "FP: ", fp
	print "TN: ", tn
	print "FN: ", fn
	if tp > 0:
		precision = tp / (tp + fp)
		recall = tp / (tp + fn)
		f1 = 2 * precision * recall / (precision + recall)
	else:
		precision = 0
		recall = 0
		f1 = 0
	print "precision: ", precision
	print "recall: ", recall
	print "f1: ", f1	



def save_to_csv(sentences, words, labeled_triggers, labeled_events, predicted_events, writefile):
	with open(writefile, 'wb') as f:
		writer = csv.writer(f)
		for i in xrange(len(sentences)):
			sentence = sentences[i]
			word = words[i]
			#labeled_trigger = labeled_triggers[i]
			labeled_event = labeled_events[i]
			predicted_event = predicted_events[i]
			tp = 0
			fp = 0
			tn = 0
			fn = 0
			#for i in range(len(predicted_event)):
			#	pred = predictions[i]
			#	label = true_labels[i]
			if labeled_event == "NONE":
				if predicted_event == "NONE":
					tn += 1
					continue
				else:
					fp += 1
			else:
				if predicted_event == labeled_event:
					tp += 1
				elif predicted_event == "NONE":
					fn += 1
				else:
					fp += 1
					fn += 1			
			writer.writerow([sentence, word, labeled_event, predicted_event, tp, fp, tn, fn])



def load_roles():
	role_file = 'argument_roles.csv'
	lines = []
	with open(role_file, 'rb') as f:
		for line in csv.reader(f):
			lines.append(line)
	lines = lines[1:]
	role_dict = {}
	for line in lines:
		event = line[0]
		role = line[1]
		types = line[2].split()
		if event not in role_dict:
			role_dict[event] = {}
		role_dict[event][role] = types
	return role_dict

def create_event_role_dict():
	role_file = 'argument_roles.csv'
	lines = []
	with open(role_file, 'rb') as f:
		for line in csv.reader(f):
			lines.append(line)
	lines = lines[1:]
	event_role_dict = {}
	for line in lines:
		event = line[0]
		role = line[1]
		if event in event_role_dict:
			event_role_dict[event].append(role)
		else:
			event_role_dict[event] = [role]
	return event_role_dict


event_role_dict = create_event_role_dict()


def create_role_type_dict():
	role_file = 'argument_roles.csv'
	lines = []
	with open(role_file, 'rb') as f:
		for line in csv.reader(f):
			lines.append(line)
	lines = lines[1:]
	event_role_dict = {}
	for line in lines:
		event = line[0]
		role = line[1]
		types = line[2].split(' ')
		if event not in event_role_dict:
			event_role_dict[event] = {}
			#event_role_dict[event].append(role)
		#else:
		#	event_role_dict[event] = {}
		event_role_dict[event][role] = types
	return event_role_dict	

event_role_type_dict = create_role_type_dict()


all_role_types = ['LOC', 'WEA', 'Title', 'MONEY', 'PER', 'CRIME', 'GPE', 'FAC', 'ORG', 'COM', 'VEH']
all_role_types = ['LOC', 'ORG', 'PER', 'GPE', 'MONEY', 'WEA', 'ART']
def get_role_types():
	return all_role_types

wn_type_to_DEFT_mapping = {}
wn_type_to_DEFT_mapping['noun.person'] = 'PER'
wn_type_to_DEFT_mapping['noun.artifact'] = 'ART'
wn_type_to_DEFT_mapping['noun.group'] = 'ORG'
wn_type_to_DEFT_mapping['noun.location'] = 'LOC'
wn_type_to_DEFT_mapping['noun.time'] = 'TIME'


all_lexnames = set()
all_lexnames.add('adj.all')
all_lexnames.add('adj.pert')
all_lexnames.add('adv.all')
all_lexnames.add('noun.Tops')
all_lexnames.add('noun.act')
all_lexnames.add('noun.animal')
all_lexnames.add('noun.artifact')
all_lexnames.add('noun.attribute')
all_lexnames.add('noun.body')
all_lexnames.add('noun.cognition')
all_lexnames.add('noun.communication')
all_lexnames.add('noun.event')
all_lexnames.add('noun.feeling')
all_lexnames.add('noun.group')
all_lexnames.add('noun.location')
all_lexnames.add('noun.motive')
all_lexnames.add('noun.object')
all_lexnames.add('noun.person')
all_lexnames.add('noun.phenomenon')
all_lexnames.add('noun.plant')
all_lexnames.add('noun.possession')
all_lexnames.add('noun.process')
all_lexnames.add('noun.quantity')
all_lexnames.add('noun.relation')
all_lexnames.add('noun.shape')
all_lexnames.add('noun.state')
all_lexnames.add('noun.substance')
all_lexnames.add('noun.time')
all_lexnames.add('verb.body')
all_lexnames.add('verb.change')
all_lexnames.add('verb.cognition')
all_lexnames.add('verb.communication')
all_lexnames.add('verb.competition')
all_lexnames.add('verb.consumption')
all_lexnames.add('verb.contact')
all_lexnames.add('verb.creation')
all_lexnames.add('verb.emotion')
all_lexnames.add('verb.perception')
all_lexnames.add('verb.possession')
all_lexnames.add('verb.social')
all_lexnames.add('verb.stative')
all_lexnames.add('verb.weather')
all_lexnames.add('adj.ppl')
#all_lexnames.add('NONE')

def get_all_lexnames():
	return all_lexnames

#def get_potential_argument_tokens_from_sentence(sentence, event):

def get_dep_path_length(dep_path):
	path_list = dep_path.split(' ')
	return min((float(len(path_list)) / 2) - 2, 0)


def get_arg_wordlist():
	return ['person', 'name', 'place', 'city', 'building', 'team', 'organization', 'time', 'clock', 'date', 'crime', 'punishment', 'weapon', 'money', 'dollars', 'beneficiary', 'recipient', 'giver', 'attacker', 'instrument', 'location', 'buyer', 'seller', '$', 'gun', 'bomb', 'company', '12:00', 'New_York', 'soldier', 'criminal']

def cosine_distance(A, B):
	A_dot_B = np.dot(A, B)
	A_norm = sqrt(np.dot(A, A))
	B_norm = sqrt(np.dot(B, B))
	return 1 - ( A_dot_B / (A_norm * B_norm) )


def get_entities(entity_tagged_sentence, sentence, dependency_type):
	entities = []
	tokens = sentence['tokens']
	this_entity = {} 
	in_entity = False
	for i, word in enumerate(entity_tagged_sentence):
		if word == 'NONE':
			if in_entity:
				entities.append(this_entity)
				this_entity = {}
				in_entity = False
		else:
			B_I = word[0]
			token = tokens[i]
			# Need to deal with cases where hits I with no B (might happen?)
			if in_entity and B_I == 'I':
				this_entity['tokens'].append(token)
				continue
			elif in_entity and B_I == 'B':
				entities.append(this_entity)
				this_entity = {}
			entity_type = word[2:]
			this_entity['tokens'] = [token]
			this_entity['type'] = entity_type
			in_entity = True
	for entity in entities:
		tokens = entity['tokens']
		head = get_head(tokens, sentence['tokens'], sentence[dependency_type])
		entity['head'] = head		
	return entities

entity_type_list = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA', 'CRIME', 'MONEY', 'Title']
#entity_type_list = ['FAC', 'GPE', 'LOC', 'ORG', 'PER', 'VEH', 'WEA']












def precision_func(tp, fp):
	if tp + fp == 0:
		return 0
	return float(tp) / (tp + fp)

def recall_func(tp, fn):
	if tp + fn == 0:
		return 0	
	return float(tp) / (tp + fn)

def f1_func(p, r):
	if p + r == 0:
		return 0
	return 2. * p * r / (p + r)

def save_event_level_preds(predictions, true_labels, savefile = False, missed_triggers=0):
	print "Doing event level predictions"
	total_counts_dict = {}
	precision_dict = {}
	recall_dict = {}
	f1_dict = {}
	event_result_dict = {}
	event_result_dict['total'] = {}
	event_result_dict['total']['tp'] = 0.
	event_result_dict['total']['fp'] = 0.
	event_result_dict['total']['tn'] = 0.
	event_result_dict['total']['fn'] = 0.
	precision_dict['total'] = 0.
	recall_dict['total'] = 0.
	f1_dict['total'] = 0.
	total_counts_dict['total'] = 0.	
	if len(predictions) != len(true_labels):
		print "Lengths don't match"
		return
	tp = 0.
	fp = 0.
	tn = 0.
	fn = float(missed_triggers)
	for i in xrange(len(predictions)):
		is_tp = False
		is_fp = False
		is_tn = False
		is_fn = False
		pred = predictions[i]
		label = true_labels[i]
		if label == "NONE":
			if pred == "NONE":
				tn += 1
				is_tn = True
			else:
				fp += 1
				is_fp = True
		else:
			if pred == label:
				tp += 1
				is_tp = True
			elif pred == "NONE":
				fn += 1
				is_fn = True
			else:
				fp += 1
				is_fp = True
				fn += 1
				is_fn = True
		if pred not in event_result_dict:
			event_result_dict[pred] = {}
			event_result_dict[pred]['tp'] = 0.
			event_result_dict[pred]['tn'] = 0.
			event_result_dict[pred]['fp'] = 0.
			event_result_dict[pred]['fn'] = 0.
		if label not in event_result_dict:
			event_result_dict[label] = {}
			event_result_dict[label]['tp'] = 0.
			event_result_dict[label]['tn'] = 0.
			event_result_dict[label]['fp'] = 0.
			event_result_dict[label]['fn'] = 0.			
		if is_tp:
			event_result_dict[pred]['tp'] += 1
		if is_fp:
			event_result_dict[pred]['fp'] += 1
		if is_tn:
			event_result_dict[pred]['tn'] += 1
		if is_fn:
			event_result_dict[label]['fn'] += 1						
	# for event, result_dict in event_result_dict.items():
	# 	p = precision(result_dict['tp'], result_dict['fp'])
	# 	r = recall(result_dict['tp'], result_dict['fn'])
	# 	event_f1 = f1_func(p, r)
	# 	if event not in precision_dict:
	# 		precision_dict[event] = 0.
	# 	if event not in recall_dict:
	# 		recall_dict[event] = 0.
	# 	if event not in f1_dict:
	# 		f1_dict[event] = 0.
	# 	precision_dict[event] += p
	# 	recall_dict[event] += r
	# 	f1_dict[event] += event_f1
	print "TP: ", tp
	print "FP: ", fp
	print "TN: ", tn
	print "FN: ", fn
	if (tp + fp) > 0:				
		precision = tp / (tp + fp)
	else:
		precision = 0
	if (tp + fn) > 0:		
		recall = tp / (tp + fn)
	else:
		recall = 0
	if (precision + recall) > 0:
		f1 = 2 * precision * recall / (precision + recall)
	else:
		f1 = 0
	print "precision: ", precision
	print "recall: ", recall
	print "f1: ", f1
	#precision_dict['total'] += precision
	#recall_dict['total'] += recall
	#f1_dict['total'] += f1
	event_result_dict['total']['tp'] += tp
	event_result_dict['total']['fp'] += fp
	event_result_dict['total']['tn'] += tn
	event_result_dict['total']['fn'] += fn		
	for event, count_dict in event_result_dict.items():
		precision = precision_func(count_dict['tp'], count_dict['fp'])
		recall = recall_func(count_dict['tp'], count_dict['fn'])
		f1 = f1_func(precision, recall)
		precision_dict[event] = precision
		recall_dict[event] = recall
		f1_dict[event] = f1
		total_counts_dict[event] = count_dict['tp'] + count_dict['fn']
	#stats_file = run_dir + 'stats_' + run_name[:len(run_name) - 4] + '.csv'	
	with open(savefile, 'wb') as f:
		writer = csv.writer(f)
		for event in precision_dict.keys():
			row = [event, precision_dict[event], recall_dict[event], f1_dict[event], total_counts_dict[event]]
			writer.writerow(row)
