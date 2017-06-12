import readEAL
import sys, os
from os import listdir
from os.path import isfile, join
import random
import csv

random.seed(4)

directory = '../../datasets/LDC2016E37_TAC_KBP_English_Event_Argument_Comprehensive_Training_and_Evaluation_Data_2014-2015/data/2015/eval/'
source_dir = directory + "source_corpus/"
manual_run_dir = directory + "manual_run/arguments/"
assessments_dir = directory + 'assessments/complete/assessment/'

nw_dir = 'nw/'
mpdf_dir = 'mpdf/'

def get_docID_from_filename(filename):
	if 'mpdf.xml' in filename:
		filename = filename.split('mpdf.xml')[0]
	elif '.xml' in filename:
		filename = filename.split('.xml')[0]
	filename = filename.split('/')
	return filename[len(filename) - 1]


def get_results_dict(path, results_files):
	results_dict = {}
	for filename in results_files:
		filename = path + filename
		doc_id = get_docID_from_filename(filename)
		results_dict[doc_id] = []
		with open(filename, 'rb') as f:
			reader = csv.reader(f, delimiter='\t')
			for line in reader:
				results_dict[doc_id].append(line)
	return results_dict


def get_manual_dict(results_dict):
	trigger_file_dict = {}
	for doc_id, list_of_extractions in results_dict.items():
		trigger_file_dict[doc_id] = []
		triggers = []
		for extraction in list_of_extractions:
			event = extraction[2]
			role = extraction[3]
			arg_text = extraction[4]
			


def get_labeled_data(annotations):
	if annotations == 'manual':
		labeled_data_dir = manual_run_dir
	else:
		labeled_data_dir = assessments_dir
	results_files = [f for f in listdir(labeled_data_dir) if isfile(join(labeled_data_dir, f))]
	if annotations == 'manual':
		label_dict = get_manual_dict(get_results_dict(labeled_data_dir, results_files))
	else:
		label_dict = get_assessment_dict(get_results_dict(labeled_data_dir, results_files))
	return label_dict


def get_KBP_data_from_source(annotations='manual'):
	#source_path = directory + source_dir
	nw_path = source_dir + nw_dir
	mpdf_path = source_dir + mpdf_dir
	nw_source_files = [f for f in listdir(nw_path) if isfile(join(nw_path, f))]
	mpdf_source_files = [f for f in listdir(mpdf_path) if isfile(join(mpdf_path, f))]
	nw_annotated_dict = readEAL.get_files_annotated_text(nw_path, nw_source_files)
	mpdf_annotated_dict = readEAL.get_files_annotated_text(mpdf_path, mpdf_source_files)
	all_annotated_dict = nw_annotated_dict
	for key, val in mpdf_annotated_dict.items():
		all_annotated_dict[key] = val
	labeled_data = get_labeled_data(annotations)
	return all_annotated_dict, labeled_data

