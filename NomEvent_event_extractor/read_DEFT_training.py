from xml.etree import ElementTree
import sys, os
from os import listdir
from os.path import isfile, join
from pycorenlp import StanfordCoreNLP
import gen_utils
import pickle

nlp = StanfordCoreNLP('http://localhost:9000')

def get_entity_dict(tree):
	entities = {}
	for node in tree.findall('.//entity_mention'):
		entity_mention_id = node.get('id')
		entity_mention_offset = node.get('offset')
		entity_mention_length = node.get('length')
		entity_mention_text = node.text
		entities[entity_mention_id] = {}
		entities[entity_mention_id]['offset'] = entity_mention_offset
		entities[entity_mention_id]['end'] = entity_mention_offset + entity_mention_length
		entities[entity_mention_id]['text'] = entity_mention_text
	return entities


def get_entities(tree):
	entities = []
	for node in tree.findall('.//entity'):
		ere_type = node.get('type')
		for mention in node.findall('entity_mention'):
			entity_mention_offset = int(mention.get('offset'))
			entity_mention_length = int(mention.get('length'))
			end = entity_mention_offset + entity_mention_length			
			full_charseq = mention.text
			offset = [entity_mention_offset, end]
			entities.append({'entity_offsets': offset, 'entity_text': full_charseq, 'type': ere_type, 'subtype': 'NA'})
	return entities	


def get_triggers(tree, entity_dict):
	triggers = []
	for node in tree.findall('.//event_mention'):
		#print node.tag, node.tail, node.attrib
		subtype = node.get('subtype')
		ere_type = node.get('type')
		uppercase_event = gen_utils.rere_lower_to_upper_mapping[ere_type + '.' + subtype]
		event_type = uppercase_event
		if event_type == "NONE":
			continue
		trigger = ""
		section_to_check = ""
		children = node.getchildren()
		args = []
		arg_texts = []
		arg_roles = []
		#txt = ""
		#offset = 0
		#length = 0
		for child in children:
			if child.tag == 'trigger':
				txt = child.text
				offset = int(child.get('offset'))
				length = int(child.get('length'))
				#triggers.append({'trigger': trigger.lower(), 'ere_type': ere_type, 'subtype': subtype, 'offset': offset, 'length': length})
				#print offset
				#section_to_check = text[max(0, offset - 200): min(len(text) - 1, offset + 200)]
				#event_count += 1
			if child.tag == 'args':
				args = []
				arg_texts = []
				arg_roles = []
				arg_children = child.getchildren()
				for arg_child in arg_children:
					arg_role = arg_child.get('type')
					if arg_role not in gen_utils.event_role_dict[event_type]:
						arg_role = arg_role[0].upper() + arg_role[1:]
						if arg_role not in gen_utils.event_role_dict[event_type]:
							if arg_role == 'Vehicle' and event_type in ['Movement.Transport-Person', 'Movement.Transport-Artifact']:
								arg_role = "Instrument"
							elif arg_role == 'Artifact' and event_type == 'Movement.Transport-Person':
								arg_type = 'Movement.Transport-Artifact'
							elif arg_role == "Buyer" and event_type == 'Transaction.Transfer-Ownership':
								arg_role = "Recipient"
							elif arg_role == "Seller" and event_type =='Transaction.Transfer-Ownership':
								arg_role = "Giver"	
							elif arg_role == "Artifact" and event_type == 'Transaction.Transfer-Ownership':
								arg_role = "Thing"
							else:						
								print arg_role, 'not in dict for', event_type
								#x = 1/0
					entity_mention_id = arg_child.get('entity_mention_id')
					arg_entity_mention = entity_dict[entity_mention_id]
					args.append([int(arg_entity_mention['offset']), int(arg_entity_mention['end'])])
					arg_texts.append(arg_entity_mention['text'])
					arg_roles.append(arg_role)
			if child.tag == 'places':
				args = []
				arg_texts = []
				arg_roles = []
				arg_children = child.getchildren()
				for arg_child in arg_children:
					arg_role = 'Place'
					entity_mention_id = arg_child.get('entity_mention_id')
					arg_entity_mention = entity_dict[entity_mention_id]
					args.append([int(arg_entity_mention['offset']), int(arg_entity_mention['end'])])
					arg_texts.append(arg_entity_mention['text'])									
		triggers.append({'trigger': txt.lower(), 'event_type': event_type, 'ere_type': ere_type, 'subtype': subtype, 'offset': offset, 'length': length,'args': args, 'arg_texts': arg_texts, 'arg_roles': arg_roles})			
	return triggers


def get_args(tree):
	args = []
	for node in tree.findall('.//event_mention'):
		#print node.tag, node.tail, node.attrib
		#subtype = node.get('subtype')
		#ere_type = node.get('type')
		#trigger = ""
		section_to_check = ""
		children = node.getchildren()
		for child in children:
			if child.tag == 'args':
				for arg in child.getchildren():
					arg_text = arg.text
					args.append(arg_text)
				#print offset
				#section_to_check = text[max(0, offset - 200): min(len(text) - 1, offset + 200)]
				#event_count += 1
	return args


def read_file(deft_directory, sourcepath, erepath, subfolder, ere_filename):
	filename = ere_filename.split('.ere.xml')[0]
	source_filename = filename + '.txt'
	ere_path = deft_directory + erepath + subfolder + ere_filename
	source_path = deft_directory + sourcepath + subfolder + source_filename
	with open(source_path, 'rb') as f:
		doc = f.read()
		doc = doc.lower()
		annotated_text = nlp.annotate(doc, properties={'timeout': 9999, 'annotators': 'tokenize, ssplit, pos, depparse, lemma, ner', 'outputFormat': 'json'})	
		annotated_text = gen_utils.get_sentence_dict(annotated_text)
	with open(ere_path, 'rb') as f:
		tree = ElementTree.parse(f)
	entities = get_entity_dict(tree)
	triggers = get_triggers(tree, entities)
	entity_list = get_entities(tree)
	return annotated_text, triggers, filename, entity_list



def get_DEFT_data_from_source():
	deft_directory = '../../DEFT-event-corpora/LDC2013E64_DEFT_Phase_1_ERE_Annotation_R3_V2/data/'
	sourcepath = 'source/'
	erepath = 'ere/'
	#subfolder = 'events_workshop/'
	#subfolder = 'osc/'
	subfolder = 'proxy/'
	mypath = deft_directory + erepath + subfolder
	files = [f for f in listdir(mypath) if isfile(join(mypath, f))]
	training_docs = []
	training_annotations = []
	training_filenames = []
	training_entities = []
	test_docs = []
	test_annotations = []
	test_filenames = []
	dev_docs = []
	dev_annotations = []
	dev_filenames = []
	for ere_filename in files:
		if 'DS_Store' in ere_filename:
			continue
		print "FILE: ", ere_filename	
		annotated_text, triggers, filename, entities = read_file(deft_directory, sourcepath, erepath, subfolder, ere_filename)
		training_docs.append(annotated_text)
		training_annotations.append(triggers)
		training_filenames.append(filename)
		training_entities.append(entities)
	to_return = {}
	to_return = {'training_docs' : training_docs, 'training_labels' : training_annotations, 'test_docs' : test_docs, 'test_labels' : test_annotations, 'dev_docs' : dev_docs, 'dev_labels' : dev_annotations, 'training_filenames' : training_filenames, 'test_filenames' : test_filenames, 'dev_filenames' : dev_filenames, 'training_entities': training_entities}
	return to_return


def get_DEFT_data(rerun_coreNLP, save_coreNLP=False):
	# 2: added entities
	deft_file = "../../datasets/deft_data2.pkl"
	if rerun_coreNLP:
		dat = get_DEFT_data_from_source()
	else:
		with open(deft_file, 'rb') as f:
			dat = pickle.load(f)
	if save_coreNLP:
		with open(deft_file, 'wb') as f:
			pickle.dump(dat, f)
	return dat
