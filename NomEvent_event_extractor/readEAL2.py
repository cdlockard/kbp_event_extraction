from xml.etree import ElementTree
import sys, os
from os import listdir
from os.path import isfile, join
from pycorenlp import StanfordCoreNLP
from xml.etree.ElementTree import iterparse
import gen_utils
import pickle
import codecs

#deft_directory = 'DEFT-event-corpora/LDC2013E64_DEFT_Phase_1_ERE_Annotation_R3_V2/data/'

sourcepath = 'source/'
erepath = 'ere/'


nlp = StanfordCoreNLP('http://localhost:9000')
#mypath = deft_directory + erepath
#files = [f for f in listdir(mypath) if isfile(join(mypath, f))]







def get_triggers(tree):
	triggers = []
	for node in tree.findall('.//event_mention'):
		#print node.tag, node.tail, node.attrib
		subtype = node.get('subtype')
		ere_type = node.get('type')
		event_type = gen_utils.rere_lower_to_upper_mapping[ere_type + '.' + subtype]
		if event_type == 'NONE':
			continue
		realis = node.get('realis')
		trigger = ""
		section_to_check = ""
		children = node.getchildren()
		arg_text = []
		arg_role = []
		arg_refids = []
		for child in children:
			if child.tag == 'trigger':
				trigger = child.text
				offset = int(child.get('offset'))
				length = int(child.get('length'))
				#triggers.append({'trigger': trigger.lower(), 'ere_type': ere_type, 'subtype': subtype, 'offset': offset, 'length': length})
				#print offset
				#section_to_check = text[max(0, offset - 200): min(len(text) - 1, offset + 200)]
				#event_count += 1
			if child.tag == 'em_arg':
				this_role = child.get('role')
				this_refid = child.get('entity_mention_id')
				if this_refid == None:
					this_refid = child.get('filler_id')
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
							print child.get('role'), 'not in role dict for ', event_type
							continue
				arg_role.append(this_role)
				arg_refids.append(this_refid)
				arg_text.append(child.text)
		triggers.append({'trigger': trigger.lower(), 'event_type': event_type, 'ere_type': ere_type.lower(), 'subtype': subtype.lower(), 'offset': offset, 'length': length, 'arg_texts': arg_text, 'arg_roles': arg_role, 'realis': realis, 'arg_refids': arg_refids})									
	return triggers



def get_entities(tree):
	entities = []
	#for node in tree.findall('.//event_mention'):
	for node in tree.findall('.//entity'):
		#print node.tag, node.tail, node.attrib
		#subtype = node.get('SUBTYPE')
		ere_type = node.get('type')
		entity_id = node.get('id')
		if 'Title' in ere_type:
			ere_type = 'Title'
		elif 'Sentence' in ere_type:
			continue
		elif 'Numeric' in ere_type:
			ere_type = 'MONEY'
		elif 'Crime' in ere_type:
			ere_type = 'CRIME'
		for mention in node.findall('entity_mention'):
			entity_mention_id = mention.get('id')
			offset = int(mention.get('offset'))
			length = int(mention.get('length'))
			arg_offset = [offset, offset + length]
			entities.append({'type': ere_type, 'entity_id': entity_id, 'entity_mention_id': entity_mention_id, 'arg_offset': arg_offset})	
		#event_type = gen_utils.ace_eal_mapping[ere_type + '.' + subtype]
		#modality = node.get('MODALITY')
		#genericity = node.get('GENERICITY')
		#realis = get_realis(modality, genericity)
		#print node.tag, ere_type, subtype
		#print subtype
		#print ere_type
		#if event_type == "NONE":
		#	continue
		# for mention in node.findall('entity_mention'):
		# 	#print mention.tag
		# 	full_charseq_start = 0
		# 	full_charseq_end = 0
		# 	#args = []
		# 	#arg_text = []
		# 	#arg_role = []
		# 	#offsets = []
		# 	#texts = []
		# 	mention_id = mention.get('ID')
		# 	for extent in mention.findall('./extent'):	
		# 		for charseq in extent.findall('./charseq'):
		# 			full_charseq = charseq.text
		# 			#full_charseq_offset_start = int(charseq.get('START'))
		# 			#full_charseq_offset_end = int(charseq.get('END'))
		# 			#full_charseq_length = full_charseq_end - full_charseq_start + 1
		# 			#offets.append([int(charseq.get('START')), int(charseq.get('END'))])
		# 			offset = [int(charseq.get('START')), int(charseq.get('END'))]
		# 			#texts.append(full_charseq)
		#	#entities.append({'entity_offsets': offset, 'entity_text': full_charseq, 'type': ere_type, 'subtype': subtype, 'mention_id': mention_id})
	for node in tree.findall('.//filler'):
		#print node.tag, node.tail, node.attrib
		#subtype = node.get('SUBTYPE')
		ere_type = node.get('type')
		filler_id = node.get('id')
		if 'title' in ere_type:
			ere_type = 'Title'
		elif 'Sentence' in ere_type:
			continue
		elif 'numeric' in ere_type:
			ere_type = 'MONEY'
		elif 'crime' in ere_type:
			ere_type = 'CRIME'
		elif 'money' in ere_type:
			ere_type = 'MONEY'
		elif ere_type == 'vehicle':
			ere_type = 'VEH'
		elif ere_type == 'weapon':
			ere_type = 'WEA'
		elif ere_type == 'commodity':
			ere_type = 'WEA'
		offset = int(node.get('offset'))
		length = int(node.get('length'))
		arg_offset = [offset, offset + length]			
		#event_type = gen_utils.ace_eal_mapping[ere_type + '.' + subtype]
		#modality = node.get('MODALITY')
		#genericity = node.get('GENERICITY')
		#realis = get_realis(modality, genericity)
		#print node.tag, ere_type, subtype
		#print subtype
		#print ere_type
		#if event_type == "NONE":
		#	continue
		# for mention in node.findall('value_mention'):
		# 	#print mention.tag
		# 	full_charseq_start = 0
		# 	full_charseq_end = 0
		# 	#args = []
		# 	#arg_text = []
		# 	#arg_role = []
		# 	#offsets = []
		# 	#texts = []
		# 	mention_id = mention.get('ID')
		# 	for extent in mention.findall('./extent'):	
		# 		for charseq in extent.findall('./charseq'):
		# 			full_charseq = charseq.text
		# 			#full_charseq_offset_start = int(charseq.get('START'))
		# 			#full_charseq_offset_end = int(charseq.get('END'))
		# 			#full_charseq_length = full_charseq_end - full_charseq_start + 1
		# 			#offets.append([int(charseq.get('START')), int(charseq.get('END'))])
		# 			offset = [int(charseq.get('START')), int(charseq.get('END'))]
		# 			#texts.append(full_charseq)
		# 			entities.append({'entity_offsets': offset, 'entity_text': full_charseq, 'type': ere_type, 'mention_id': mention_id})					
		entities.append({'type': ere_type, 'entity_mention_id': filler_id, 'arg_offset': arg_offset})						
	return entities





def get_files_annotated_text(mypath, files):
	texts = []
	for filename in files:
		if 'DS_Store' in filename:
			continue
		print "FILE: ", filename
		#ere_file = mypath + filename
		#with open(ere_file) as fd:
		#	tree = ElementTree.parse(fd)
		#sourcefile = filename.split('.ere')[0] + '.txt'
		original_text_filepath = mypath + filename
		#with open(original_text_filepath) as f:
		with codecs.open(original_text_filepath, encoding='utf-8') as f:
			text = f.read()
		text = text.encode(errors='replace')	
		#stripped_text = gen_utils.replace_tags_with_spaces(text)	
		#annotated_text = nlp.annotate(stripped_text, properties={'timeout': 9999, 'annotators': 'tokenize, ssplit, pos, depparse, lemma, ner', 'outputFormat': 'json'})
		annotated_text = nlp.annotate(text, properties={'timeout': 9999, 'annotators': 'tokenize, ssplit, pos, depparse, lemma, ner', 'outputFormat': 'json'})
		annotated_text = gen_utils.get_sentence_dict(annotated_text)
		#texts[filename] = annotated_text
		texts.append(annotated_text)
		#triggers = get_triggers(tree)
		#args = get_args(tree)	
	return texts

def get_files_triggers(mypath, files):
	all_triggers = []
	for filename in files:
		if 'DS_Store' in filename:
			continue
		print "FILE: ", filename
		if '.mpdf.xml' in filename:
			filename = filename.split('.mpdf.xml')[0]
		elif '.xml' in filename:
			filename = filename.split('.xml')[0]
		if 'AFP_ENG' in filename or 'APW_ENG' in filename or 'XIN_ENG' in filename:
			filename = filename + '-kbp'
		filename = filename + '.rich_ere.xml'
		ere_file = mypath + filename
		with open(ere_file) as fd:
			tree = ElementTree.parse(fd)
		#sourcefile = filename.split('.ere')[0] + '.txt'
		#original_text_filepath = deft_directory + sourcepath + sourcefile
		#with open(original_text_filepath) as f:
		#	text = f.read()
		#stripped_text = replace_SGML_with_space(text)	
		#annotated_text = nlp.annotate(stripped_text, properties={'timeout': 9999, 'annotators': 'tokenize, ssplit, pos, depparse', 'outputFormat': 'json'})
		#text[filename] = annotated_text
		triggers = get_triggers(tree)
		entities = get_entities(tree)
		arg_id_dict = create_arg_id_type_dict(entities)
		for trigger in triggers:
			arg_types = []
			args = []
			for j, arg_refid in enumerate(trigger['arg_refids']):
				if arg_refid in arg_id_dict:
					arg_type, arg_offset = arg_id_dict[arg_refid]
					arg_types.append(arg_type)
					args.append(arg_offset)
				else:
					print 'could not find arg type for :', filename, arg_refid, trigger['arg_roles'][j]
					print '***&*&*&&*******&&&&&&&*********'
					arg_types.append('NONE')
					args.append('NONE')
					continue
			trigger['arg_types'] = arg_types
			trigger['args'] = args
		all_triggers.append(triggers)
		#args = get_args(tree)	
	return all_triggers	


def create_arg_id_type_dict(entities):
	id_type_dict = {}
	for entity in entities:
		#print entity
		id_type_dict[entity['entity_mention_id']] = [entity['type'], entity['arg_offset']]
	return id_type_dict

def get_EAL_data(eal_directory, rerun_coreNLP, save_coreNLP=False):
	#eal_file = "../../datasets/eal_data.pkl"
	# for version 3, added ability to use triggers with arg types
	# actually, just added the labels in general
	#eal_file = "../../datasets/eal_data3.pkl"
	# for version 4, updated with new CoreNLP Dep types
	eal_file = "../../datasets/eal_data4.pkl"
	if rerun_coreNLP:
		dat = get_EAL_data_from_source(eal_directory)
	else:
		with open(eal_file, 'rb') as f:
			dat = pickle.load(f)
	if save_coreNLP:
		with open(eal_file, 'wb') as f:
			pickle.dump(dat, f)
	return dat	

def get_filenames(source_files):
	filenames = []
	for filename in source_files:
		if 'DS_Store' in filename:
			continue
		if '.mpdf.xml' in filename:
			filename = filename.split('.mpdf.xml')[0]
		elif '.xml' in filename:
			filename = filename.split('.xml')[0]
		filenames.append(filename)
	filenames.sort()	
	return filenames	

def get_EAL_data_from_source(eal_directory):
	sourcepath = 'source/'
	erepath = 'ere/'
	source_dir = eal_directory + sourcepath
	ere_dir = eal_directory + erepath
	source_files = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]
	filenames = get_filenames(source_files)
	prefixes = ['AFP_ENG', 'APW_ENG', 'NYT_ENG', 'XIN_ENG']
	annotated_texts = get_files_annotated_text(source_dir, source_files)
	labels = get_files_triggers(ere_dir, source_files)
	to_return = {}
	to_return['training_docs'] = annotated_texts
	to_return['training_labels'] = labels
	to_return['training_filenames'] = filenames
	return to_return
