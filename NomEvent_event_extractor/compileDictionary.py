import sys, os, csv
import nltk
from nltk.corpus import wordnet as wn
from nltk.corpus import framenet as fn
import gensim
import scipy
import math


model = gensim.models.word2vec.Word2Vec.load_word2vec_format(os.path.join(os.path.dirname("__file__"), 'GoogleNews-vectors-negative300.bin'), binary=True)

pickle_noms = True
save_csv = False


w2v = True
ms_threshold = 0.75
all_wn_synsets = False
first_fn_match = False
hyponyms = True
use_framenet = True
auto_wn_synset = False
synset_percent_threshold = 0.5
use_synset_percent_threshold = False
allow_synset_NA_counts = True
require_both_wn_an_fn = False
add_verbs = True



#writefile = 'nominals_may12/testing_all_nominals_WN_FN_hyponyms_syn_percent_noNA_requireWNFN.csv'
#savefile = 'nominals/forJames/default-with_verbs'
savefile = 'nominals/nomsApril2017/default-with_verbs'
writefile = savefile + '.csv'
pickle_file = savefile + '.pkl'

synset_dict = {}
synset_dict['meet'] = [1,2,3,8, 13]
synset_dict['attack']= [0, 9, 14] # 12?
synset_dict['die'] = [3]
synset_dict['buy'] = [1] # maybe more
synset_dict['elect'] = [1]

#synset_dict['meet'] = [0]
#synset_dict['attack']= [0] # 12?
#synset_dict['die'] = [0]
#synset_dict['buy'] = [0] # maybe more
#synset_dict['elect'] = [0]

synset_dict['bankrupt'] = [0]
synset_dict['close'] = [5]
synset_dict['merge'] = [2]
synset_dict['found'] = [1,2]
synset_dict['demonstrate'] = [3]
synset_dict['broadcast'] = [0]
synset_dict['correspond'] = [2]
synset_dict['acquit'] = [0]
synset_dict['appeal'] = [2,4]
synset_dict['arrest'] = [0,2]
synset_dict['indict'] = [0]
synset_dict['convict'] = [2]
synset_dict['execute'] = [0]
synset_dict['extradite'] = [0]
synset_dict['fine'] = [0,1]
synset_dict['pardon'] = [0,1,2,4]
synset_dict['parole'] = [2,3]
synset_dict['sentence'] = [1,2,3]
synset_dict['sue'] = [1]
synset_dict['hearing'] = [0,8]
synset_dict['birth'] = [0,1,2,4,5]
synset_dict['divorce'] = [0,2]
synset_dict['injure'] = [0,2]
synset_dict['marry'] = [0,1]
synset_dict['manufacture'] = [0,1,2,5]
#synset_dict['move'] = [6]
synset_dict['move'] = [5]
synset_dict['quit'] = [1]
synset_dict['nominate'] = [0,1]
synset_dict['hire'] = [1,2]


event_verb_mapping = {}
event_verb_mapping['elect'] = 'PersonnelEvent.Elect'
event_verb_mapping['pardon'] = 'JusticeEvent.Pardon'
event_verb_mapping['sentence'] = 'JusticeEvent.Sentencing'
event_verb_mapping['move'] = 'Movement.Transport.TransportArtifact'
event_verb_mapping['move'] = 'Movement.Transport.TransportPerson'
event_verb_mapping['hearing'] = 'JusticeEvent.TrialHearing'
event_verb_mapping['indict'] = 'JusticeEvent.ChargeIndict'
event_verb_mapping['close'] = 'BusinessEvent.EndOrganization'
event_verb_mapping['fine'] = 'JusticeEvent.Fine'
event_verb_mapping['quit'] = 'PersonnelEvent.EndPosition'
event_verb_mapping['sue'] = 'JusticeEvent.Sue'
event_verb_mapping['acquit'] = 'JusticeEvent.Acquit'
event_verb_mapping['bankrupt'] = 'BusinessEvent.DeclareBankruptcy'
event_verb_mapping['extradite'] = 'JusticeEvent.Extradite'
event_verb_mapping['divorce'] = 'LifeEvent.Divorce'
event_verb_mapping['arrest'] = 'JusticeEvent.ArrestJail'
event_verb_mapping['attack'] = 'Conflict.Attack'
event_verb_mapping['convict'] = 'JusticeEvent.Convict'
event_verb_mapping['appeal'] = 'JusticeEvent.Appeal'
event_verb_mapping['manufacture'] = 'Manufacture.Artifact'
event_verb_mapping['buy'] = 'Transaction.TransferOwnership'
event_verb_mapping['execute'] = 'JusticeEvent.Execute'
event_verb_mapping['broadcast'] = 'Contact.Broadcast'
event_verb_mapping['birth'] = 'LifeEvent.BeBorn'
event_verb_mapping['hire'] = 'PersonnelEvent.StartPosition'
event_verb_mapping['die'] = 'LifeEvent.Die'
event_verb_mapping['marry'] = 'LifeEvent.Marry'
event_verb_mapping['injure'] = 'LifeEvent.Injure'
event_verb_mapping['merge'] = 'BusinessEvent.MergeOrg'
event_verb_mapping['nominate'] = 'PersonnelEvent.Nominate'
event_verb_mapping['demonstrate'] = 'Conflict.Demonstrate'
event_verb_mapping['parole'] = 'JusticeEvent.ReleaseParole'
event_verb_mapping['meet'] = 'Contact.Meet'
event_verb_mapping['correspond'] = 'Contact.Correspondence'
event_verb_mapping['found'] = 'BusinessEvent.StartOrganization'






frame_dict = {}
#frame_dict['meet'] = [0]
#frame_dict['attack'] = [0]
#frame_dict['die'] =  [0]
#frame_dict['buy'] = [0]
#frame_dict['elect'] = [0]


frame_dict['merge'] = ['Cause_to_amalgamate']
frame_dict['found'] = ['Intentionally_create']
frame_dict['attack'] = ['Attack']
frame_dict['demonstrate'] = ['Protest']
frame_dict['meet'] = ["Discussion"]
frame_dict['arrest'] = ["Arrest"]
frame_dict['indict'] = ["Notification_of_charges"]
frame_dict['execute'] = ["Execution"]
frame_dict['extradite'] = ["Extradition"]
frame_dict['fine'] = ["Fining"]
frame_dict['pardon'] = ["Pardon"]
frame_dict['sentence'] = ["Sentencing"]
frame_dict['trial'] = ["Trial"]
frame_dict['birth'] = ["Birth", "Giving_birth"]
frame_dict['die'] = ["Death"]
frame_dict['injure'] = ["Cause_harm"]
frame_dict['manufacture'] = ["Manufacturing"]
frame_dict['quit'] = ["Quitting"]
frame_dict['buy'] = ["Commerce_buy"]
frame_dict['elect'] = ["Change_of_leadership"]


if all_wn_synsets:
	for word in synset_dict.keys():
		synset_dict[word] = range(0, len(wn.synsets(word)))
		#synset_dict[word] = [0]

if auto_wn_synset:
	for word in synset_dict.keys():
		synsets = wn.synsets(word)
		#sense2freq = {}
		max_count = 0
		max_synset = 0
		for i, s in enumerate(synsets):
			freq = 0  
			for lemma in s.lemmas():
				freq+=lemma.count()
			if freq > max_count:
				max_count = freq
				max_synset = i
			#sense2freq[i] = freq
		synset_dict[word] = [max_synset]


if first_fn_match and use_framenet:
	for word in synset_dict.keys():
		if len(fn.frames_by_lemma(word)) > 0:
			#print word
			frame_dict[word] = [fn.frames_by_lemma(word)[0].name]#map(lambda x: x.name, fn.frames_by_lemma(word)[0])




###############################


wordcount_filename = 'WordNet-InfoContent-3.0/ic-brown-add1.dat'
lines = []

with open(wordcount_filename) as f:
	lines = f.readlines()

word_id_dict = {}
for line in lines[1:]:
	line = line.split(' ')
	wordId = line[0]
	count = line[1]
	word_id_dict[wordId] = count


syns = list(wn.all_synsets())
offsets_list = [(s.offset(), s) for s in syns]
offsets_dict = dict(offsets_list)


word_dict = {}
for word_Id, count in word_id_dict.items():
	try:
		#print word_Id[:-1]
		word = offsets_dict[int(word_Id[:-1])]
		#print word
		word_dict[word] = count
	except KeyError:
		continue

for word, count in word_dict.items():
	if count[-1] == '\n':
		count = count[:-1]
	try:
		word_dict[word] = int(count)
	except:
		print word, count

################################























###############################

#word_list = ['meet', 'attack', 'die', 'buy', 'elect']
word_list = synset_dict.keys()



def get_relative_synset_frequencies(word):
	global synset_lemma_counts
	global synset_lemma_totals	
	if word in synset_lemma_totals:
		return
	synset_lemma_counts[word] = {}
	synset_lemma_totals[word] = 0
	for syn in wn.synsets(word):
		lemmas = syn.lemmas()
		for lemma in lemmas:
			if word in lemma.name() and lemma.synset().pos() == 'n' and lemma.synset() not in synset_lemma_counts[word]:
				#print 'adding synset ', lemma.synset(), lemma.count()
				synset_lemma_counts[word][lemma.synset()] = lemma.count()
				synset_lemma_totals[word] += lemma.count()


synset_lemma_counts = {}
synset_lemma_totals = {}
for word in word_list:
	get_relative_synset_frequencies(word)



def get_initial_feature_dict(word, event):
	features = {}
	features['word'] = word
	features['event'] = event
	features['FN'] = 0
	features['num_wordnet'] = 0

def add_word(word, event, features_dict):
	global reverse_dict
	if word in reverse_dict:
		if event in reverse_dict[word]:
			existing_features = reverse_dict[word][event]
			if features_dict['fn'] == 1:
				existing_features['fn'] = 1
			if math.isnan(existing_features['num_wordnet']):
				existing_features['num_wordnet'] = features_dict['num_wordnet']
			elif not math.isnan(features_dict['num_wordnet']):
				existing_features['num_wordnet'] += features_dict['num_wordnet']
			if math.isnan(existing_features['synset_percent']):
				existing_features['synset_percent'] = features_dict['synset_percent']
			elif not math.isnan(features_dict['synset_percent']):
				existing_features['synset_percent'] += features_dict['synset_percent']				
		else:
			reverse_dict[event] = features_dict
	else:
		reverse_dict[word] = {}
		reverse_dict[word][event] = features_dict


reverse_dict = {}
nominal_dict = {}
for word in word_list:
	event = event_verb_mapping[word]
	nominal_dict[word] = []
	nominals = {}
	#added_nouns = set()
	#frames = fn.frames_by_lemma(word)
	if use_framenet and word in frame_dict:
		for frame_index in frame_dict[word]:
			if len(fn.frames(frame_index)) < 1:
				print "no frame for: ", frame_index
				continue
			frame = fn.frames(frame_index)[0]
			for potential_noun in frame.lexUnit.keys():
				lemma = potential_noun.split('.')[0]
				pos = potential_noun.split('.')[1]
				if pos == 'n' or add_verbs:
					features_dict = {}
					features_dict['event'] = event
					features_dict['word'] = lemma
					features_dict['pos'] = pos
					#features_dict['synset'] = float('nan')	
					features_dict['fn'] = 1
					features_dict['num_wordnet'] = 0
					features_dict['synset_percent'] = 0
					if w2v and lemma in model and word in model:
						features_dict['w2v_distance'] = scipy.spatial.distance.cosine(model[word], model[lemma])
					else:
						features_dict['w2v_distance'] = 1					
					add_word(lemma, event, features_dict)		
					nominals[lemma] = ([lemma, ['fn'], [frame_index], ['fn'], ['fn'], ['fn'], float('nan'), []])
	synsets = wn.synsets(word)
	for synset_index in synset_dict[word]:
		synset = synsets[synset_index]
		lemmas = synset.lemmas()
		synset_count = 0
		for lemma in lemmas:
			synset_count += lemma.count()
		if hyponyms:
			for hyponym in synset.hyponyms():
				lemmas += hyponym.lemmas()
				print "ADDING HYPONYMS:", word, hyponym.lemmas()
		for lemma in lemmas:
			try:
				if w2v == False or scipy.spatial.distance.cosine(model[word], model[lemma.name()]) < ms_threshold:
					related_forms = lemma.derivationally_related_forms()
					for related_form in related_forms + [lemma]:
						if related_form.synset().pos() == "n" or add_verbs:
							if related_form.name() not in synset_lemma_totals:
								get_relative_synset_frequencies(related_form.name())
							if synset_lemma_totals[related_form.name()] == 0:
								synset_percent = 0
							else:
								synset_percent = float(synset_lemma_counts[related_form.name()][related_form.synset()]) / synset_lemma_totals[related_form.name()]
							features_dict = {}
							features_dict['event'] = event
							features_dict['word'] = related_form.name()
							features_dict['pos'] = related_form.synset().pos()
							#features_dict['synset'] = float('nan')	
							features_dict['fn'] = 0
							features_dict['num_wordnet'] = 1
							features_dict['synset_percent'] = synset_percent
							if w2v and lemma.name() in model and word in model:
								features_dict['w2v_distance'] = scipy.spatial.distance.cosine(model[word], model[lemma.name()])
							else:
								features_dict['w2v_distance'] = 1
							add_word(related_form.name(), event, features_dict)
							if related_form.name() not in nominals:
								nominals[related_form.name()] = [related_form.name(), ['wn'], [synset_index], [lemma.count()], [synset_count], [word_dict[synset]], synset_percent, [related_form.synset()]]
							else:
								if nominals[related_form.name()][6] == float('nan'):
									nominals[related_form.name()][6] = 0
									nominals[related_form.name()][7] = []
								if synset_lemma_totals[related_form.name()] != 0 and related_form.synset() not in nominals[related_form.name()][7]:
									nominals[related_form.name()][6] += synset_percent
								nominals[related_form.name()][1].append('wn')
								nominals[related_form.name()][2].append(synset_index)
								nominals[related_form.name()][3].append(lemma.count())
								nominals[related_form.name()][4].append(synset_count)
								nominals[related_form.name()][5].append(word_dict[synset])
								#nominals[related_form.name()][6].append(synset_percent)
								nominals[related_form.name()][7].append(related_form.synset())
			except KeyError:
				print "key error", word, lemma.name()	
	for key, val in nominals.items():
		nominal_dict[word].append(val)	



if pickle_noms:
	import pickle
	with open(pickle_file, 'wb') as f:
		pickle.dump(reverse_dict, f)


if save_csv:
	with open(writefile, 'wb') as f:
		writer = csv.writer(f)
		for key, vals in nominal_dict.items(): # in nominalizations.keys():
			for val in vals:
				try:
					cos_sim = 0
					if w2v:
						cos_sim = scipy.spatial.distance.cosine(model[key], model[val[0]])
					if cos_sim <= ms_threshold:
						if val[6] == 'NA' and not allow_synset_NA_counts:
							continue
						if require_both_wn_an_fn and ('wn' not in val[1] or 'fn' not in val[1]):
							continue
						if not use_synset_percent_threshold or val[6] == 'nan' or val[6] >= synset_percent_threshold:
							writer.writerow([event_verb_mapping[key], val[0], "%.2f" % float(val[6]), "%.4f" % cos_sim])
				except KeyError:
					print "key error", key, val


