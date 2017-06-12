import csv
import nltk
import os


def get_event_verb_mapping():
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
	return event_verb_mapping


def get_nominals_list(filename):
	event_verb_mapping = get_event_verb_mapping()
	nominals_list = []
	with open(filename) as f:
		for r in csv.reader(f):
			nominals_list.append(r)
	reverse_dict = {}
	for line in nominals_list:
		#verb, noun, source, indices, a, b, c, synset_percent, score = line
		verb = line[0]
		if verb in event_verb_mapping:
			event = event_verb_mapping[verb]
		else:
			event = verb
		noun = line[1]
		#if verb not in events_to_search:
		#	continue
		if noun in reverse_dict:
			reverse_dict[noun].append(event)
		else:
			reverse_dict[noun] = [event]	
	for key, val in reverse_dict.items():
		reverse_dict[key + 's'] = val			
	return reverse_dict	

def get_noun_types():
	noun_types = ['NN', 'NN$', 'NNS', 'NNP', 'NNPS']
	return noun_types

def get_verb_types():
	verb_types = ['VB', 'VBD', 'VBG', 'VBN', 'VBP', 'VBZ']	
	return verb_types

def get_event_dict(reverse_dict):
	event_dict = {}
	for key, vals in reverse_dict.items():
		for val in vals:
			if val in event_dict:
				event_dict[val].append(key)
			else:
				event_dict[val] = [key]	
	return event_dict