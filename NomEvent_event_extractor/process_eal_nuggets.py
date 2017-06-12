import readEAL
import extraction_utils
import nom_utils
from os import listdir
from os.path import isfile, join
import os
import pickle
import gensim
import random
import pattern_utils2
import time
#import parallel_utils3
import parallel_utils_nuggets
import gen_utils
import gc

# Read in data


nug_data = ''


save_dir = ''

if not os.path.exists(save_dir):
    os.makedirs(save_dir)


raw_text_directory = '/projects/WebWare6/KBP_2016/LDC2016E64_TAC_KBP_2016_Evaluation_Core_Source_Corpus/data/eng/'

nominals_dir = 'nominals/may20/'
nominals_file = 'default.csv'



reduce_deps = True
rerun_coreNLP = False
resave_coreNLP = False
check_all_noms = True
testing = False
re_gather_deps = False
check_only_nouns = False
parallelize_compute = True
use_full_charseq = False
use_w2v = True
use_lemmas = True

num_cores = 1



dependency_type = 'collapsed-ccprocessed-dependencies'
noms_filename = 'nominals/forJames/default-with_verbs.pkl'


nominals_filename = nominals_dir + nominals_file

reverse_dict = nom_utils.get_nominals_list(nominals_filename)
event_dict = nom_utils.get_event_dict(reverse_dict)

high_recall_noms_filename = 'nominals/forJames/default-with_verbs.csv'
high_recall_wordlist = nom_utils.get_nominals_list(high_recall_noms_filename)
for word, events in reverse_dict.items():
	if word not in high_recall_wordlist:
		print word, events
		high_recall_wordlist[word] = events


event_dict = nom_utils.get_event_dict(reverse_dict)


def get_data(number):
	filename = 'datasets/eal/small_tranch' + str(number) + '.pkl'
	with open(filename, 'rb') as f:
		docs = pickle.load(f)
	return docs

with open(nug_data, 'rb') as f:
	docs = pickle.load(f)

docs = docs['training_docs']



print "Loaded Data..."

random.seed(4)

# Get classifier

trigger_classifier_filename = '../../output/september29/lr_model_all_training_all_wordsFIXED.pkl'

# COMPETITION

classifier_dict_filename = '../../output/september1_forCompetition_allACEplusEAL/all_words.csvarg_classifier_dict.pkl'


realis_classifier_filename = '../../output/august23/lr_model_realis_classifier_wouldaVerbNegNerWouldacount_noLSH.pkl'




crf_file = 'crf_testACEDEFT150_090116.crfsuite'
# Get deps
with open('../../person_deps_all_words_10.pkl', 'rb') as f:
	#with open('../../person_deps.pkl', 'rb') as f:
	person_deps = pickle.load(f)

with open('../../arg_deps_all_words_20.pkl', 'rb') as f:
	#with open('../../arg_deps.pkl', 'rb') as f:
	real_arg_deps = pickle.load(f)			


bigram_set = set()
arg_deps = []
noun_deps = []
wn_types_features = gen_utils.get_all_lexnames()



print 'loading w2v'
w2v_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(os.path.join(os.path.dirname("__file__"), '../../GoogleNews-vectors-negative300.bin'), binary=True)
print 'w2v loaded'


start_time = time.time()

parallel_utils_nuggets.parallel_call_to_extract_from_docs_and_save_with_args(docs.values(), docs.keys(), save_dir, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, real_arg_deps, wn_types_features, trigger_classifier_filename, classifier_dict_filename, realis_classifier_filename, raw_text_directory, crf_file, w2v_model)





end_time = time.time()

print str(float(end_time - start_time) / 60) + " Minutes"

print save_dir

print "completed"