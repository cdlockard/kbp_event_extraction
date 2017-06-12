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
		realis = node.get('realis')
		trigger = ""
		section_to_check = ""
		children = node.getchildren()
		for child in children:
			if child.tag == 'trigger':
				trigger = child.text
				offset = int(child.get('offset'))
				length = int(child.get('length'))
				triggers.append({'trigger': trigger.lower(), 'ere_type': ere_type, 'subtype': subtype, 'offset': offset, 'length': length})
				#print offset
				#section_to_check = text[max(0, offset - 200): min(len(text) - 1, offset + 200)]
				#event_count += 1
	return triggers

def get_files_annotated_text(mypath, files):
	texts = {}
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
		stripped_text = gen_utils.replace_tags_with_spaces(text)	
		annotated_text = nlp.annotate(stripped_text, properties={'timeout': 9999, 'annotators': 'tokenize, ssplit, pos, depparse, lemma, ner', 'outputFormat': 'json'})
		annotated_text = gen_utils.get_sentence_dict(annotated_text)
		texts[filename] = annotated_text
		#triggers = get_triggers(tree)
		#args = get_args(tree)	
	return texts

def get_files_triggers(mypath, files):
	all_triggers = {}
	for filename in files:
		if 'DS_Store' in filename:
			continue
		print "FILE: ", filename
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
		all_triggers[filename] = triggers
		#args = get_args(tree)	
	return all_triggers	


def get_EAL_data(eal_directory, rerun_coreNLP, save_coreNLP=False):
	#eal_file = "../../datasets/eal_data.pkl"
	eal_file = "../../datasets/eal_data2.pkl"
	if rerun_coreNLP:
		dat = get_EAL_data_from_source(eal_directory)
	else:
		with open(eal_file, 'rb') as f:
			dat = pickle.load(f)
	if save_coreNLP:
		with open(eal_file, 'wb') as f:
			pickle.dump(dat, f)
	return dat	

def get_EAL_data_from_source(eal_directory):
	sourcepath = 'source/'
	source_dir = eal_directory + sourcepath
	source_files = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]
	prefixes = ['AFP_ENG', 'APW_ENG', 'NYT_ENG', 'XIN_ENG']
	annotated_texts = get_files_annotated_text(source_dir, source_files)
	return annotated_texts
