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
import parallel_utils3
import arg_utils

## Options
run_dir = '../../output/august23/'
if not os.path.exists(run_dir):
    os.makedirs(run_dir)


#run_name = 'ACE_8_4_ace4_with_deps10_timexLabels_hashedBG_NOnounPatterns.csv'
#run_name = "testing_for_args_bg10_actually_working_for_real.csv"
run_name = 'realis_classifier_wouldaVerbNegNerWouldacount_noLSH.csv'

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
check_only_nouns = False
parallelize_compute = True
use_full_charseq = False
use_w2v = True
use_lemmas = False
do_args_too = True

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
#DEFTdat = read_DEFT_training.get_DEFT_data(True, True)

training_docs = dat['training_docs']
training_trigger_info = dat['training_labels']
training_filenames = dat['training_filenames']
#print len(training_docs)
train_set_indices = range(len(training_docs))

dev_docs = dat['dev_docs']
dev_trigger_info = dat['dev_labels']
dev_filenames = dat['dev_filenames']
dev_set_indices = range(len(dev_docs))

testing_docs = dat['test_docs']
testing_trigger_info = dat['test_labels']
testing_filenames = dat['test_filenames']
test_set_indices = range(len(testing_docs))



print "Read in data"
## At some point we have all docs and trigger_labels



## Harvest deps:
arg_deps = []
noun_deps = []
person_deps = []
bigram_set = set()
real_arg_deps = []

if re_gather_deps:
	print "Gathering Deps"
	#arg_deps = gen_utils.gather_arg_deps(training_docs, training_trigger_info, train_set_indices, reduce_deps, dependency_type, 2)
	if parallelize_compute:
		#noun_deps = gen_utils.gather_deps_par(training_docs, training_trigger_info, train_set_indices, reduce_deps, dependency_type, parallelize_compute, check_only_nouns, 10, gen_utils.get_doc_noun_deps, num_cores)
		print "noun deps: " + str(len(noun_deps))
		person_deps = gen_utils.gather_deps_par(training_docs, training_trigger_info, train_set_indices, reduce_deps, dependency_type, parallelize_compute, check_only_nouns, 10, gen_utils.get_doc_person_deps, num_cores)
		print "person deps: " + str(len(person_deps))
		real_arg_deps = gen_utils.gather_deps_par(training_docs, training_trigger_info, train_set_indices, reduce_deps, dependency_type, parallelize_compute, check_only_nouns, 100, gen_utils.get_doc_trigger_deps, num_cores)
		print 'argument deps: ' + str(len(real_arg_deps))
		#bigram_set = gen_utils.gather_word_bigrams(training_docs, use_lemmas, parallelize_compute, num_cores)
		with open('../../person_deps.pkl', 'wb') as f:
			pickle.dump(person_deps, f)
		with open('../../arg_deps.pkl', 'wb') as f:
			pickle.dump(real_arg_deps, f)			
	else:
		noun_deps = gen_utils.gather_noun_deps(training_docs, training_trigger_info, train_set_indices, reduce_deps, dependency_type, parallelize_compute, 50)
		person_deps = gen_utils.gather_person_deps(training_docs, training_trigger_info, train_set_indices, reduce_deps, dependency_type, parallelize_compute, 75)
		#bigram_set = gen_utils.gather_word_bigrams(training_docs, use_lemmas, parallelize_compute, num_cores)
else:
	with open('../../person_deps.pkl', 'rb') as f:
		person_deps = pickle.load(f)	
	with open('../../arg_deps.pkl', 'rb') as f:
		real_arg_deps = pickle.load(f)			


print "Got deps, creating tf..."

## Create tf:
w2v_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(os.path.join(os.path.dirname("__file__"), '../../GoogleNews-vectors-negative300.bin'), binary=True)
#w2v_model = gensim.models.word2vec.Word2Vec.load_word2vec_format(os.path.join(os.path.dirname("__file__"), 'GoogleNews-vectors-negative300.bin'), binary=True)
#w2v_model = {}
#token_featurizer = pattern_utils2.TokenFeaturizer(dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, w2v_model, False)



#def process_parallel(dependency_type, reduce_deps, token_featurizer, classifier, parallelize_compute, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, training=True):
#	token_featurizer = pattern_utils2.TokenFeaturizer(dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, w2v_model, False)	
#	classifier = LogisticRegression()
#	extractor = extraction_utils.EventExtractor(dependency_type, reduce_deps, token_featurizer, classifier, parallelize_compute, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, training=True)


#def create_parallel_system():
#	docs_per_core = num_docs / num_cores

print "Created tf, starting feature gathering"
## Create event extractor:
parallelize_compute = True
#num_cores = 20
#parallel_utils2.set_pool(num_cores)
classifier = LogisticRegression()
#extractor = extraction_utils.EventExtractor(dependency_type, reduce_deps, token_featurizer, classifier, parallelize_compute, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, num_cores, training=True)

if parallelize_compute:
	#training_examples = parallel_utils.parallel_call_to_process_docs(extractor, training_docs, training_trigger_info, training_filenames, num_cores)
	training_examples = parallel_utils3.parallel_call_to_process_docs(training_docs, training_trigger_info, training_filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, w2v_model, True)
else:	
	training_examples = extractor.process_docs(training_docs, training_trigger_info, training_filenames)

training_features, training_labels, training_sentences, training_words, training_labeled_triggers = extraction_utils.get_corpus_info_realis(training_examples)


print "Total time after feature processing for " + run_name + ": " + str(float((time.time() - start_time)) / 60 ) + " minutes"
print "Finished processing features, starting training... "

lr = classifier
lr.fit(training_features, training_labels)
#lr.score(training_features, training_labels)
training_pred = lr.predict(training_features)
print "Total time after training " + run_name + ": " + str(float((time.time() - start_time)) / 60 ) + " minutes"
print 'Training results for ' + run_name
gen_utils.compute_precision_recall(training_pred, training_labels)

model_file = run_dir + 'lr_model_' + run_name[:len(run_name) - 4] + '.pkl'
#model_file = 'lr_model_all_noms_8_3_all_words.pkl'
#tf_file = 'tf_model.pkl'
#training_data_file = 'train_data_8_2.pkl'

# Save

with open(model_file, 'wb') as f:
	pickle.dump(lr, f)



print "Starting dev:"
# Dev:
#num_cores = 5
missed_triggers = 0
if parallelize_compute:
	#dev_examples = parallel_utils2.parallel_call_to_process_docs(extractor, dev_docs[:10], dev_trigger_info[:10], dev_filenames[:10], num_cores)
	dev_examples = parallel_utils3.parallel_call_to_process_docs(dev_docs, dev_trigger_info, dev_filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, w2v_model, True)
else:
	dev_examples = extractor.process_docs(dev_docs[:10], dev_trigger_info[:10], dev_filenames[:10])

dev_features, dev_labels, dev_sentences, dev_words, dev_labeled_triggers = extraction_utils.get_corpus_info_realis(dev_examples)

#dev_token_features, dev_raw_labels, dev_texts = process_docs(training_docs, training_trigger_info, training_filenames, dev_set_indices)
#dev_features, dev_labels = compile_features_labels(dev_token_features, dev_raw_labels)
dev_pred = lr.predict(dev_features)
proba = lr.predict_proba(dev_features)
#precision_recall_fscore_support(dev_labels, pred)
print 'results for ' + run_name
lr_name = run_dir + 'lr_results_dev' + run_name
gen_utils.compute_precision_recall(dev_pred, dev_labels, lr_name)
#compute_precision_recall(pred, dev_labels)
gen_utils.compute_precision_recall_proba(proba, dev_labels, lr, 0.3)

gen_utils.save_to_csv(dev_sentences, dev_words, dev_labeled_triggers, dev_labels, dev_pred, writefile)


print "Starting test:"

# Test:
#num_cores = 5
missed_triggers = 0
if parallelize_compute:
	#test_examples = parallel_utils2.parallel_call_to_process_docs(extractor, testing_docs, testing_trigger_info, testing_filenames, num_cores)
	test_examples = parallel_utils3.parallel_call_to_process_docs(testing_docs, testing_trigger_info, testing_filenames, num_cores, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, arg_deps, noun_deps, person_deps, bigram_set, high_recall_wordlist, event_dict, check_only_nouns, check_all_noms, w2v_model, True)
else:
	test_examples = extractor.process_docs(testing_docs, testing_trigger_info, testing_filenames)

test_features, test_labels, test_sentences, test_words, test_labeled_triggers = extraction_utils.get_corpus_info_realis(test_examples)

#dev_token_features, dev_raw_labels, dev_texts = process_docs(training_docs, training_trigger_info, training_filenames, dev_set_indices)
#dev_features, dev_labels = compile_features_labels(dev_token_features, dev_raw_labels)
pred = lr.predict(test_features)
proba = lr.predict_proba(test_features)
#precision_recall_fscore_support(dev_labels, pred)
print 'results for ' + run_name
lr_name = run_dir + 'lr_results_test' + run_name
gen_utils.compute_precision_recall(pred, test_labels, lr_name)
#compute_precision_recall(pred, dev_labels)
gen_utils.compute_precision_recall_proba(proba, test_labels, lr, 0.3)
test_writefile = run_dir + run_name[:len(run_name) - 4] + '_testSetOutput.csv'
gen_utils.save_to_csv(test_sentences, test_words, test_labeled_triggers, test_labels, pred, test_writefile)

run_rf = True
if run_rf:
	print "Trying Random Forest:"
	from sklearn.ensemble import RandomForestClassifier
	clf = RandomForestClassifier(n_estimators=500)
	clf.fit(training_features, training_labels)
	pred = clf.predict(test_features)
	#rf_name = run_dir + 'rf_results' + run_name
	gen_utils.compute_precision_recall(pred, test_labels)
	#compute_precision_recall(pred, dev_labels)

print "Total time for " + run_name + ": " + str(float((time.time() - start_time)) / 60 ) + " minutes before args"

