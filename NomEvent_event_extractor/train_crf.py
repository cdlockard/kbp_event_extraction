import gen_utils
import extraction_utils
import nom_utils
import pattern_utils2
from sklearn.linear_model import LogisticRegression
import readACE5
import readEAL
import read_DEFT_training
import time
import os, csv
import gensim
import pickle
import parallel_utils5
import arg_utils
import pycrfsuite
from itertools import chain
import nltk
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelBinarizer
import sklearn
import crf_utils


## Options
run_dir = '../../output/august26/'
if not os.path.exists(run_dir):
    os.makedirs(run_dir)


run_name = 'ACE_nouns_only.csv'


start_time = time.time()


nominals_dir = 'nominals/may20/'
nominals_file = 'default.csv'

#deft_directory = 'DEFT-event-corpora/LDC2013E64_DEFT_Phase_1_ERE_Annotation_R3_V2/data/'
deft_directory = '../../datasets/LDC2016E60_TAC_KBP_2016_English_Event_Argument_Linking_Pilot_Gold_Standard/data/'

sourcepath = 'source/'
erepath = 'ere/'


writefile = run_dir + run_name



reduce_deps = True
rerun_coreNLP = False
resave_coreNLP = False
check_all_noms = True
testing = False
re_gather_deps = False
check_only_nouns = True
parallelize_compute = True
use_full_charseq = False
use_w2v = True
use_lemmas = True
do_args_too = True
use_DEFT_for_args = False

num_cores = 20



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





## Read in files

# Get ACE

dat = readACE5.get_ACE_data(rerun_coreNLP, resave_coreNLP)
DEFTdat = read_DEFT_training.get_DEFT_data(False, False)






training_docs = dat['training_docs']
training_trigger_info = dat['training_labels']
training_filenames = dat['training_filenames']
training_entities = dat['training_entities']
print len(training_docs)
training_docs += DEFTdat['training_docs']
training_trigger_info += DEFTdat['training_labels']
training_filenames += DEFTdat['training_filenames']
training_entities += DEFTdat['training_entities']
print len(training_docs)
train_set_indices = range(len(training_docs))

dev_docs = dat['dev_docs']
dev_trigger_info = dat['dev_labels']
dev_filenames = dat['dev_filenames']
dev_entities = dat['dev_entities']
dev_set_indices = range(len(dev_docs))

testing_docs = dat['test_docs']
testing_trigger_info = dat['test_labels']
testing_filenames = dat['test_filenames']
testing_entities = dat['test_entities']
test_set_indices = range(len(testing_docs))

print "Read in data"
## At some point we have all docs and trigger_labels



def bio_classification_report(y_true, y_pred):
    """
    Classification report for a list of BIO-encoded sequences.
    It computes token-level metrics and discards "O" labels.
    
    Note that it requires scikit-learn 0.15+ (or a version from github master)
    to calculate averages properly!
    """
    lb = LabelBinarizer()
    y_true_combined = lb.fit_transform(list(chain.from_iterable(y_true)))
    y_pred_combined = lb.transform(list(chain.from_iterable(y_pred)))
        
    tagset = set(lb.classes_) - {'NONE'}
    tagset = sorted(tagset, key=lambda tag: tag.split('-', 1)[::-1])
    class_indices = {cls: idx for idx, cls in enumerate(lb.classes_)}
    
    return classification_report(
        y_true_combined,
        y_pred_combined,
        labels = [class_indices[cls] for cls in tagset],
        target_names = tagset,
    )


#w2v_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(os.path.join(os.path.dirname("__file__"), '../../GoogleNews-vectors-negative300.bin'), binary=True)


#def train():
training_features, training_labels = crf_utils.process_docs(training_docs, training_entities)
test_features, test_labels = crf_utils.process_docs(testing_docs, testing_entities)

trainer = pycrfsuite.Trainer(verbose=True)

for xseq, yseq in zip(training_features, training_labels):
    trainer.append(xseq, yseq)

trainer.set_params({
    'c1': 1.0,   # coefficient for L1 penalty
    'c2': 1e-3,  # coefficient for L2 penalty
    'max_iterations': 150,  # stop earlier

    # include transitions that are possible, but not observed
    'feature.possible_transitions': True
})

crf_file = 'crf_testACEDEFT150_090116.crfsuite'
trainer.train(crf_file)
tagger = pycrfsuite.Tagger()
tagger.open(crf_file)
y_pred = [tagger.tag(xseq) for xseq in test_features]
print bio_classification_report(test_labels, y_pred)

y_pred = [tagger.tag(xseq) for xseq in training_features]
print bio_classification_report(training_labels, y_pred)

