from xml.etree import ElementTree
import sys, os
from os import listdir
from os.path import isfile, join
from pycorenlp import StanfordCoreNLP
from xml.etree.ElementTree import iterparse
import gen_utils
import pickle
import random
import codecs

random.seed(4)
#deft_directory = 'DEFT-event-corpora/LDC2013E64_DEFT_Phase_1_ERE_Annotation_R3_V2/data/'

sourcepath = 'source/'
erepath = 'ere/'


nlp = StanfordCoreNLP('http://localhost:9000')
#mypath = deft_directory + erepath
#files = [f for f in listdir(mypath) if isfile(join(mypath, f))]







def get_files_annotated_text(files):
	texts = {}
	for filename in files:
		if 'DS_Store' in filename:
			print 'encountered DS_Store'
			continue			
		print "FILE: ", filename
		#ere_file = mypath + filename
		#with open(ere_file) as fd:
		#	tree = ElementTree.parse(fd)
		#sourcefile = filename.split('.ere')[0] + '.txt'
		#original_text_filepath = mypath + filename
		original_text_filepath = filename
		#with open(original_text_filepath) as f:
		with codecs.open(original_text_filepath, encoding='utf-8') as f:
			text = f.read()
		#print type(text)
		#print isinstance(text, str)
		#text = str(text)
		text = text.encode(errors='replace')
		stripped_text = gen_utils.replace_tags_with_spaces(text)	
		annotated_text = nlp.annotate(stripped_text, properties={'timeout': 9999, 'annotators': 'tokenize, ssplit, pos, depparse, lemma, ner', 'outputFormat': 'json'})
		#print annotated_text
		annotated_text = gen_utils.get_sentence_dict(annotated_text)
		filename_key = filename.split('/')[-1]
		texts[filename_key] = annotated_text
		#triggers = get_triggers(tree)
		#args = get_args(tree)	
	return texts



def get_EAL_data(eal_filename):
	#eal_file = "../../datasets/ace_data12.pkl"
	#eal_file = "../../datasets/eal/" + eal_filename + '.pkl'
	with open(eal_file, 'rb') as f:
		dat = pickle.load(f)
	return dat	

def save_EAL_data_pickle(eal_files, savefile):
	eal_file = "../../datasets/nugget/" + savefile + '.pkl'
	dat = get_EAL_data_from_source(eal_files)
	with open(eal_file, 'wb') as f:
		pickle.dump(dat, f)


def get_EAL_data_from_source(eal_files):
	#sourcepath = 'source/'
	#source_dir = eal_directory + sourcepath
	#source_files = [f for f in listdir(source_dir) if isfile(join(source_dir, f))]
	#prefixes = ['AFP_ENG', 'APW_ENG', 'NYT_ENG', 'XIN_ENG']
	annotated_texts = get_files_annotated_text(eal_files)
	return annotated_texts





def read_all_filenames():
	parent_directory = 'datasets/LDC2016E64_TAC_KBP_2016_Evaluation_Core_Source_Corpus/data/eng/'
	nw_dir = parent_directory + 'nw/'
	df_dir = parent_directory +'df/'
	#all_files = map(lambda x: df_dir + x + '.xml' if 'ENG_DF' in x else nw_dir + x + '.xml', files_to_find)
	nw_files = [f for f in listdir(nw_dir) if isfile(join(nw_dir, f))]
	nw_files = map(lambda x: nw_dir + x, nw_files)
	df_files = [f for f in listdir(df_dir) if isfile(join(df_dir, f))]
	df_files = map(lambda x: df_dir + x, df_files)
	all_files = nw_files + df_files
	#random.shuffle(all_files)
	#all_files = [f for f in listdir(parent_directory) if isfile(join(parent_directory, f))]
	#all_files = map(lambda x: parent_directory + x, all_files)
	num_tranches = 1
	num_per_tranch = len(all_files) / num_tranches
	tranches = []
	for i in range(num_tranches):
		start = i * num_per_tranch
		if i < (num_tranches - 1):
			end = start + num_per_tranch
		else:
			end = len(all_files)
		this_tranch = all_files[start:end]
		tranches.append(this_tranch)
	# confirm length
	total_length = 0
	for tranch in tranches:
		total_length += len(tranch)
	assert total_length == len(all_files)
	for i, tranch in enumerate(tranches):
		print 'starting tranch ', str(i)
		save_EAL_data_pickle(tranch, 'full_parse_nugget_2016_' + str(i))	
		print 'finished tranch ', str(i)



read_all_filenames()

