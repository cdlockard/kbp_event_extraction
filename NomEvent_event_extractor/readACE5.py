from xml.etree import ElementTree
import sys, os
from os import listdir
from os.path import isfile, join
from pycorenlp import StanfordCoreNLP
from xml.etree.ElementTree import iterparse
import gen_utils
import pickle
import codecs

nlp = StanfordCoreNLP('http://localhost:9000')


ace_folder = '../../ACE2005/data/English/'

folders = ['bc/', 'bn/', 'cts/', 'nw/', 'un/', 'wl/']

use_timex = False
if use_timex:
	adj_folder = 'adj/'
else:
	adj_folder = 'timex2norm/'


def get_realis(modality, genericity):
	if genericity == 'Generic':
		return 'Generic'
	if modality == 'Asserted':
		return 'Actual'
	return 'Other'


def get_triggers(tree):
	triggers = []
	#for node in tree.findall('.//event_mention'):
	for node in tree.findall('.//event'):
		#print node.tag, node.tail, node.attrib
		subtype = node.get('SUBTYPE')
		ere_type = node.get('TYPE')
		event_type = gen_utils.ace_eal_mapping[ere_type + '.' + subtype]
		modality = node.get('MODALITY')
		genericity = node.get('GENERICITY')
		realis = get_realis(modality, genericity)
		#print node.tag, ere_type, subtype
		#print subtype
		#print ere_type
		if event_type == "NONE":
			continue
		for mention in node.findall('event_mention'):
			#print mention.tag
			full_charseq_start = 0
			full_charseq_end = 0
			args = []
			arg_text = []
			arg_role = []
			arg_refids = []
			for extent in mention.findall('./extent'):
				for charseq in extent.findall('./charseq'):
					full_charseq = charseq.text
					full_charseq_offset_start = int(charseq.get('START'))
					full_charseq_offset_end = int(charseq.get('END'))
					full_charseq_length = full_charseq_end - full_charseq_start + 1
			for arg in mention.findall('./event_mention_argument'):
				this_role = arg.get('ROLE')
				this_refid = arg.get('REFID')
				if this_role not in gen_utils.event_role_dict[event_type]:
					this_role = this_role[0].upper() + this_role[1:]
					if this_role not in gen_utils.event_role_dict[event_type]:
						if this_role == 'Vehicle' and event_type in ['Movement.Transport-Person', 'Movement.Transport-Artifact']:
							this_role = "Instrument"
						elif this_role == 'Artifact' and event_type == 'Movement.Transport-Person':
							event_type = 'Movement.Transport-Artifact'
						elif this_role == "Buyer" and event_type == 'Transaction.Transfer-Ownership':
							this_role = "Recipient"
						elif this_role == "Seller" and event_type =='Transaction.Transfer-Ownership':
							this_role = "Giver"	
						elif this_role == "Artifact" and event_type == 'Transaction.Transfer-Ownership':
							this_role = "Thing"	
						elif this_role == "Victim" and event_type == 'Conflict.Attack':
							this_role = 'Target'
						elif this_role == 'Entity' and event_type == 'Personnel.Elect':
							this_role = 'Person'
						elif this_role == 'Person' and event_type == 'Life.Die':
							this_role = 'Victim'
						elif this_role == 'Agent' and event_type == 'Conflict.Attack':
							this_role = 'Attacker'
						else:
							print arg.get('ROLE'), 'not in role dict for ', event_type
							continue
				arg_role.append(this_role)
				arg_refids.append(this_refid)
				for charseq in arg.findall('.//charseq'):
					args.append([int(charseq.get('START')), int(charseq.get('END'))])
					arg_text.append(charseq.text)
			for anchor in mention.findall('./anchor'):
				for charseq in anchor.findall('./charseq'):
					#print charseq.text
					txt = charseq.text
					start = int(charseq.get('START'))
					end = int(charseq.get('END'))
					length = end - start + 1
					triggers.append({'trigger': txt.lower(), 'event_type': event_type, 'ere_type': ere_type.lower(), 'subtype': subtype.lower(), 'offset': start, 'length': length, 'full_charseq': full_charseq, 'full_charseq_offset_start': full_charseq_offset_start, 'full_charseq_length': full_charseq_length,'args': args, 'arg_texts': arg_text, 'arg_roles': arg_role, 'realis': realis, 'arg_refids': arg_refids})
	return triggers


def get_entities(tree):
	entities = []
	#for node in tree.findall('.//event_mention'):
	for node in tree.findall('.//entity'):
		#print node.tag, node.tail, node.attrib
		subtype = node.get('SUBTYPE')
		ere_type = node.get('TYPE')
		if 'Title' in ere_type:
			ere_type = 'Title'
		elif 'Sentence' in ere_type:
			continue
		elif 'Numeric' in ere_type:
			ere_type = 'MONEY'
		elif 'Crime' in ere_type:
			ere_type = 'CRIME'
		#event_type = gen_utils.ace_eal_mapping[ere_type + '.' + subtype]
		#modality = node.get('MODALITY')
		#genericity = node.get('GENERICITY')
		#realis = get_realis(modality, genericity)
		#print node.tag, ere_type, subtype
		#print subtype
		#print ere_type
		#if event_type == "NONE":
		#	continue
		for mention in node.findall('entity_mention'):
			#print mention.tag
			full_charseq_start = 0
			full_charseq_end = 0
			#args = []
			#arg_text = []
			#arg_role = []
			#offsets = []
			#texts = []
			mention_id = mention.get('ID')
			for extent in mention.findall('./extent'):	
				for charseq in extent.findall('./charseq'):
					full_charseq = charseq.text
					#full_charseq_offset_start = int(charseq.get('START'))
					#full_charseq_offset_end = int(charseq.get('END'))
					#full_charseq_length = full_charseq_end - full_charseq_start + 1
					#offets.append([int(charseq.get('START')), int(charseq.get('END'))])
					offset = [int(charseq.get('START')), int(charseq.get('END'))]
					#texts.append(full_charseq)
					entities.append({'entity_offsets': offset, 'entity_text': full_charseq, 'type': ere_type, 'subtype': subtype, 'mention_id': mention_id})
	for node in tree.findall('.//value'):
		#print node.tag, node.tail, node.attrib
		#subtype = node.get('SUBTYPE')
		ere_type = node.get('TYPE')
		if 'Title' in ere_type:
			ere_type = 'Title'
		elif 'Sentence' in ere_type:
			continue
		elif 'Numeric' in ere_type:
			ere_type = 'MONEY'
		elif 'Crime' in ere_type:
			ere_type = 'CRIME'		
		#event_type = gen_utils.ace_eal_mapping[ere_type + '.' + subtype]
		#modality = node.get('MODALITY')
		#genericity = node.get('GENERICITY')
		#realis = get_realis(modality, genericity)
		#print node.tag, ere_type, subtype
		#print subtype
		#print ere_type
		#if event_type == "NONE":
		#	continue
		for mention in node.findall('value_mention'):
			#print mention.tag
			full_charseq_start = 0
			full_charseq_end = 0
			#args = []
			#arg_text = []
			#arg_role = []
			#offsets = []
			#texts = []
			mention_id = mention.get('ID')
			for extent in mention.findall('./extent'):	
				for charseq in extent.findall('./charseq'):
					full_charseq = charseq.text
					#full_charseq_offset_start = int(charseq.get('START'))
					#full_charseq_offset_end = int(charseq.get('END'))
					#full_charseq_length = full_charseq_end - full_charseq_start + 1
					#offets.append([int(charseq.get('START')), int(charseq.get('END'))])
					offset = [int(charseq.get('START')), int(charseq.get('END'))]
					#texts.append(full_charseq)
					entities.append({'entity_offsets': offset, 'entity_text': full_charseq, 'type': ere_type, 'mention_id': mention_id})					
	return entities


def read_sgm(filename):
	#with open(filename, 'rb') as f:
	with codecs.open(filename, encoding='utf-8') as f:	
		dat = f.read()
	dat = dat.encode(errors='replace')
	return dat

def read_annotated(filename):
	with open(filename) as f:
		tree = ElementTree.parse(f)
	return get_triggers(tree)

def read_annotated_entities(filename):
	with open(filename) as f:
		tree = ElementTree.parse(f)
	return get_entities(tree)	

def get_filenames(ace_folder, folder, adj_folder):
	mypath = ace_folder + folder + adj_folder
	files = [join(mypath, f) for f in listdir(mypath) if (isfile(join(mypath, f)) and '.DS_Store' not in f)]
	files.sort()
	return files


def get_date(filename):
	with open(filename, 'rb') as f:
		dat = f.readlines()
	if dat[3][15] == '-':
		return dat[3][11:15] + dat[3][16:18] + dat[3][19:21]
	return dat[3][11:19]

def get_file_offset_adjusts(filename):
	#with open(filename, 'rb') as f:
	with codecs.open(filename, encoding='utf-8') as f:	
		dat = f.read()
	dat = dat.encode(errors='replace')		
	open_tag = False	
	remove_count = 0
	tag_ends = []
	start_seg = 0
	for i, character in enumerate(dat):
		if character == '<':
			open_tag = True
			start_seg = i
			#if char_count > 2:
			#	if dat[i - 2] != '>':
			#		remove_count = 1
		elif character == '>':
			open_tag = False
			#remove_count += 1
			tag_ends.append((start_seg, remove_count + 1))			
			remove_count = 0
		if open_tag:
			remove_count += 1
	return tag_ends

def get_offset_adjust(begin, end, offset_adjusts):
	total_offset = 0
	total_chars = 0
	for offset_adjust in offset_adjusts:
		adjust_begin = offset_adjust[0]
		adjust_end = adjust_begin + offset_adjust[1]
		if total_offset + begin > adjust_begin:
			#continue
			total_offset += offset_adjust[1]
	return total_offset


def create_arg_id_type_dict(entities):
	id_type_dict = {}
	for entity in entities:
		#print entity
		id_type_dict[entity['mention_id']] = entity['type']
	return id_type_dict


def get_ACE_data_from_source(james_split = True):
	docs = []
	annotations = []
	dates = []
	folder_rec = []
	filenames_rec = []
	offset_adjusts = []	
	entities = []
	for folder in folders:
		filenames = get_filenames(ace_folder, folder, adj_folder)
		for filename in filenames:
			if '.apf.xml' in filename and 'score' not in filename:
				#print filename
				annotations.append(read_annotated(filename))
				folder_rec.append(folder)
				entities.append(read_annotated_entities(filename))
				arg_id_type_dict = create_arg_id_type_dict(entities[-1])
				for trigger in annotations[-1]:
					arg_types = []
					for j, arg_refid in enumerate(trigger['arg_refids']):
						role = trigger['arg_roles'][j]
						event_type = trigger['event_type']
						if arg_refid in arg_id_type_dict:
							arg_types.append(arg_id_type_dict[arg_refid])
							arg_type = arg_id_type_dict[arg_refid]
							if role.lower() == 'artifact':
								if 'movement' in event_type.lower() and arg_type.lower() == 'per':
									trigger['event_type'] = 'Movement.Transport-Person'
									trigger['arg_roles'][j] = 'Person'
						else:
							print 'arg type not found', filename, trigger['arg_roles'][j]
							arg_types.append('NONE')
					trigger['arg_types'] = arg_types
				#filenames_rec.append(filename)
			elif '.sgm' in filename:
				#print filename
				#doc = gen_utils.replace_tags_with_spaces(read_sgm(filename))
				doc = gen_utils.replace_tags_and_metadata_with_spaces(read_sgm(filename))
				annotated_text = nlp.annotate(doc, properties={'timeout': 9999, 'annotators': 'tokenize, ssplit, pos, depparse, lemma, ner', 'outputFormat': 'json'})	
				annotated_text = gen_utils.get_sentence_dict(annotated_text)
				docs.append(annotated_text)
				dates.append(get_date(filename))
				folder_rec.append(folder)
				filenames_rec.append(filename)
				offset_adjusts.append(get_file_offset_adjusts(filename))
				#print dates[-1]
			else:
				continue	
	#fixed_annotations = []
	assert len(annotations) == len(entities) == len(docs)
	for i, annotation in enumerate(annotations):
		#fixed_triggers = []
		for trigger in annotation:
			begin = trigger['offset']
			end = begin + trigger['length']
			adjust = get_offset_adjust(begin, end, offset_adjusts[i])
			trigger['offset'] = trigger['offset'] + adjust
			## then do full charseq
			begin = trigger['full_charseq_offset_start']
			end = begin + trigger['full_charseq_length']
			full_charseq_adjust = get_offset_adjust(begin, end, offset_adjusts[i])
			trigger['full_charseq_offset_start'] = trigger['full_charseq_offset_start'] + full_charseq_adjust
			## then do args:
			new_args = []
			for arg in trigger['args']:
				arg_offset_adjust = get_offset_adjust(arg[0], arg[1], offset_adjusts[i])
				new_arg = [arg[0] + arg_offset_adjust, arg[1] + arg_offset_adjust]
				new_args.append(new_arg)
			trigger['args'] = new_args
			#fixed_annotations.append(annotation[0] + adjust, annotation[1])
	for i, annotation in enumerate(entities):
		for entity in annotation:
			offsets = entity['entity_offsets']
			begin = offsets[0]
			end = offsets[1]
			#print 'old offsets:', str(begin), str(end)
			adjust = get_offset_adjust(begin, end, offset_adjusts[i])
			begin = begin + adjust
			end = end + adjust
			#print 'new offsets:', str(begin), str(end)
			entity['entity_offsets'] = [begin, end]
			#print 'new offsets', str(entity['entity_offsets'])
	#annotations = fixed_annotations
	training_docs = []
	training_annotations = []
	training_folders = []
	training_dates = []
	training_filenames = []
	training_offset_adjusts = []
	training_entities = []
	test_docs = []
	test_annotations = []
	test_folders = []
	test_dates = []
	test_filenames = []
	test_offset_adjusts = []
	test_entities = []
	dev_docs = []
	dev_annotations = []
	dev_folders = []
	dev_dates = []
	dev_filenames = []
	dev_offset_adjusts = []	
	dev_entities = []
	#train test split:
	training_year = {}
	training_month = {}
	training_year['nw/'] = 2003
	training_year['bn/'] = 2003
	training_year['bc/'] = 2003
	training_year['wl/'] = 2005
	training_year['un/'] = 2005
	training_year['cts/'] = 2004
	# train/test split months:
	training_month['nw/'] = 6
	training_month['bn/'] = 6
	training_month['bc/'] = 6
	training_month['wl/'] = 2
	training_month['un/'] = 2
	training_month['cts/'] = 11
	if james_split:
		james_training_filenames, james_dev_filenames, james_test_filenames = get_james_split()
		for i, doc in enumerate(docs):
			filename = filenames_rec[i]
			if '../../' in filename:
				filename = filename.split('../../')[1]
			if filename in james_training_filenames:
				training_docs.append(doc)
				training_annotations.append(annotations[i])
				training_folders.append(folder_rec[i])
				training_dates.append(dates[i])
				training_filenames.append(filenames_rec[i])
				training_offset_adjusts.append(offset_adjusts[i])
				training_entities.append(entities[i])
			elif filename in james_dev_filenames:
				dev_docs.append(doc)
				dev_annotations.append(annotations[i])
				dev_folders.append(folder_rec[i])
				dev_dates.append(dates[i])
				dev_filenames.append(filenames_rec[i])
				dev_offset_adjusts.append(offset_adjusts[i])
				dev_entities.append(entities[i])
			elif filename in james_test_filenames:
				test_docs.append(doc)
				test_annotations.append(annotations[i])
				test_folders.append(folder_rec[i])
				test_dates.append(dates[i])
				test_filenames.append(filenames_rec[i])
				test_offset_adjusts.append(offset_adjusts[i])	
				test_entities.append(entities[i])
			else:
				print 'Filename Not Found: ', filename	
	else:		
		for i, doc in enumerate(docs):
			date_year = int(dates[i][:4])
			date_month = int(dates[i][4:6])
			date_day = int(dates[i][6:])
			#print date_year, date_month, folder_rec[i]
			if date_year <= training_year[folder_rec[i]] and date_month <= training_month[folder_rec[i]]:
				training_docs.append(doc)
				training_annotations.append(annotations[i])
				training_folders.append(folder_rec[i])
				training_dates.append(dates[i])
				training_filenames.append(filenames_rec[i])
				training_offset_adjusts.append(offset_adjusts[i])
				training_entities.append(entities[i])
			else:
				test_docs.append(doc)
				test_annotations.append(annotations[i])
				test_folders.append(folder_rec[i])
				test_dates.append(dates[i])
				test_filenames.append(filenames_rec[i])
				test_offset_adjusts.append(offset_adjusts[i])
				test_entities.append(entities[i])
	to_return = {'training_docs' : training_docs, 'training_labels' : training_annotations, 'test_docs' : test_docs, 'test_labels' : test_annotations, 'dev_docs' : dev_docs, 'dev_labels' : dev_annotations, 'training_filenames' : training_filenames, 'test_filenames' : test_filenames, 'dev_filenames' : dev_filenames, 'training_entities': training_entities, 'dev_entities': dev_entities, 'test_entities': test_entities}
	return to_return

def get_ACE_data(rerun_coreNLP, save_coreNLP=False):
	#ace 8: modified to use EAL types only
	#ace 9: changed phone-write to map to correspondence rather than Broadcast
	# ace 10: added entities, changed to read unicode and convert with replacement
	# ace 11: added arg entity type
	# ace 12: changed transport-artifact to be transport-person if artifact is a PER type
	# ace 13: running with new version of corenlp (with new plusplus dependencies)
	ace_file = "../../datasets/ace_data13.pkl"
	if rerun_coreNLP:
		dat = get_ACE_data_from_source()
	else:
		with open(ace_file, 'rb') as f:
			dat = pickle.load(f)
	if save_coreNLP:
		with open(ace_file, 'wb') as f:
			pickle.dump(dat, f)
	return dat

def get_james_split():
	split_folder = '../../ace_splits/'
	training_file = 'new_filelist_ACE_training.txt'
	dev_file = 'new_filelist_ACE_dev.txt'
	test_file = 'new_filelist_ACE_test.txt'	
	with open(split_folder + training_file, 'rb') as f:
		training_filenames = f.readlines()
	with open(split_folder + dev_file, 'rb') as f:
		dev_filenames = f.readlines()
	with open(split_folder + test_file, 'rb') as f:
		test_filenames = f.readlines()
	if not use_timex:
		training_filenames = map(lambda x: 'ACE2005/data/English/' + x.split('\n')[0] + '.sgm', training_filenames)
		dev_filenames = map(lambda x: 'ACE2005/data/English/' + x.split('\n')[0] + '.sgm', dev_filenames)
		test_filenames = map(lambda x: 'ACE2005/data/English/' + x.split('\n')[0] + '.sgm', test_filenames)
	else:
		training_filenames = map(lambda x: x.split('timex2norm')[0] + 'adj' + x.split('timex2norm')[1], training_filenames)
		dev_filenames = map(lambda x: x.split('timex2norm')[0] + 'adj' + x.split('timex2norm')[1], dev_filenames)	
		test_filenames = map(lambda x: x.split('timex2norm')[0] + 'adj' + x.split('timex2norm')[1], test_filenames)
	return [training_filenames, dev_filenames, test_filenames]