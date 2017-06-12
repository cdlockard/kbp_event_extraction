import gen_utils

# code adapted from http://sklearn-crfsuite.readthedocs.io/en/latest/tutorial.html

type_map = {}
type_map['Agent'] = "Person"
type_map['Artifact'] = "Thing"
type_map['Attacker'] = "Person"
type_map['Beneficiary'] = "Person"
type_map['Crime'] = "Crime"
type_map['Destination'] = "Place"
type_map['Entity'] = "Person"
type_map['Giver'] = "Person"
type_map['Instrument'] = "Thing"
type_map['Money'] = "Thing"
type_map['NONE'] = "NONE"
type_map['Origin'] = "Place"
type_map['Person'] = "Person"
type_map['Place'] = "Place"
type_map['Position'] = "Thing"
type_map['Recipient'] = "Person"
type_map['Target'] = "Person"
type_map['Thing'] = "Thing"
type_map['Victim'] = "Person"



type_map = {}
type_map['Agent'] = "Entity"
type_map['Artifact'] = "Entity"
type_map['Attacker'] = "Entity"
type_map['Beneficiary'] = "Entity"
type_map['Crime'] = "Entity"
type_map['Destination'] = "Entity"
type_map['Entity'] = "Entity"
type_map['Giver'] = "Entity"
type_map['Instrument'] = "Entity"
type_map['Money'] = "Entity"
type_map['NONE'] = "NONE"
type_map['Origin'] = "Entity"
type_map['Person'] = "Entity"
type_map['Place'] = "Entity"
type_map['Position'] = "Entity"
type_map['Recipient'] = "Entity"
type_map['Target'] = "Entity"
type_map['Thing'] = "Entity"
type_map['Victim'] = "Entity"





def word2features(sent, i):
    token = sent['tokens'][i]
    word = token['word']
    postag = token['pos']
    nertag = token['ner']
    features = [
        'bias',
        'word.lower=' + word.lower(),
        'word[-3:]=' + word[-3:],
        'word[-2:]=' + word[-2:],
        'word.isupper=%s' % word.isupper(),
        'word.istitle=%s' % word.istitle(),
        'word.isdigit=%s' % word.isdigit(),
        'postag=' + postag,
        'postag[:2]=' + postag[:2],
        'nertag=' + nertag,
    ]
    if i > 0:
        token1 = sent['tokens'][i-1]
        word1 = token1['word']
        postag1 = token1['pos']
        nertag1 = token1['ner']
        #wn_type1 = gen_utils.get_wn_type(word1)
        features.extend([
            '-1:word.lower=' + word1.lower(),
            '-1:word.istitle=%s' % word1.istitle(),
            '-1:word.isupper=%s' % word1.isupper(),
            '-1:postag=' + postag1,
            '-1:postag[:2]=' + postag1[:2],
            '-1:nertag=' + nertag1,
        ])
    else:
        features.append('BOS')       
    if i < len(sent['tokens'])-1:
        token1 = sent['tokens'][i+1]
        word1 = token1['word']
        postag1 = token1['pos']
        nertag1 = token1['ner']
        #wn_type1 = gen_utils.get_wn_type(word1)
        features.extend([
            '+1:word.lower=' + word1.lower(),
            '+1:word.istitle=%s' % word1.istitle(),
            '+1:word.isupper=%s' % word1.isupper(),
            '+1:postag=' + postag1,
            '+1:postag[:2]=' + postag1[:2],
            '+1:nertag=' + nertag1,
        ])
    else:
        features.append('EOS')
    return features



def process_token(token, sentence):
	sentence_index = token['index'] - 1
	features = word2features(sentence, sentence_index)
	return features


def process_sentence(sentence, entity_labels):
	role_tokens = []
	sentence_features = []
	sentence_labels = []
	num_events = 0
	for token in sentence['tokens']:
		token_entities = []
	prev_role = 'NONE'
	for token in sentence['tokens']:
		token_role = "NONE"
		token_entities = gen_utils.get_entities_for_token(token, entity_labels)
		if len(token_entities) > 0:
			token_role = token_entities[0]['type']
		#for j, role in enumerate(arg_roles):
		#	arg_tokens = role_tokens[j]
		#	if token in arg_tokens:
		#		token_role = type_map[role]
		features = process_token(token, sentence)
		sentence_features.append(features)
		#if token_role == 'NONE':
		#	if token['ner'] != 'O':
		#		token_role = 'Entity'
		if token_role == 'TITLE':
			token_role = 'Title'
		#elif token_role == 'Contact-Info':
		#	token_role = 'NONE'
		if token_role != "NONE":
			if prev_role == 'NONE' or token_role not in prev_role:
				token_role = 'B_' + token_role
			else:
				token_role = 'I_' + token_role
		prev_role = token_role
		sentence_labels.append(token_role)		
	return sentence_features, sentence_labels


def process_doc(doc, entity_info):
	doc_features = []
	doc_labels = []
	for sentence in doc['sentences']:
		sentence_features, sentence_labels = process_sentence(sentence, entity_info)
		doc_features.append(sentence_features)
		doc_labels.append(sentence_labels)
	return doc_features, doc_labels

def process_docs(docs, entity_infos):
	all_features = []
	all_labels = []
	for i, doc in enumerate(docs):
		if i % 20 == 0:
			print float(i) / len(docs)
		entity_info = entity_infos[i]
		doc_features, doc_labels = process_doc(doc, entity_info)
		all_features += doc_features
		all_labels += doc_labels
	return all_features, all_labels
