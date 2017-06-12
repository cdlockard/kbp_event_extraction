import gen_utils
import nom_utils
from pattern_utils2 import TokenFeaturizer

#import scipy

class ArgFeaturizer(TokenFeaturizer):
	def __init__(self, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, dep_pattern_features, noun_dep_pattern_features, person_dep_pattern_features, bigram_feature_set, w2v_model, trigger_dep_pattern_features, wn_types_features, testing=False):
		TokenFeaturizer.__init__(self, dependency_type, reduce_deps, use_lemmas, use_w2v, noms_filename, dep_pattern_features, noun_dep_pattern_features, person_dep_pattern_features, bigram_feature_set, w2v_model, testing)
		self.trigger_dep_pattern_features = trigger_dep_pattern_features
		self.wn_types_features = wn_types_features
		self.arg_wordlist = gen_utils.get_arg_wordlist()
		self.prev_word_match_list = ['the', 'a', "'s", 'of', 'to']
		self.next_word_match_list = ["'s", 'of', 'to']
	def construct_features(self, arg_token, trigger_token, sentence, events, doc_event_word_counts, matching_dep_patterns, matching_noun_dep_patterns, matching_person_dep_patterns, matching_trigger_dep_patterns, back_bigram, forward_bigram, hashed_sentence, entity_type, entity_tagged_sentence):
		hashed_sentence = []
		base_features = TokenFeaturizer.construct_features(self, arg_token, sentence, events, doc_event_word_counts, matching_dep_patterns, matching_noun_dep_patterns, matching_person_dep_patterns, back_bigram, forward_bigram, hashed_sentence, entity_tagged_sentence)
		matching_trigger_dep_patterns = [str(matching_trigger_dep_patterns[0]) + " wn " + 'wn' + ' word ' + 'word']
		#trigger_path = gen_utils.find_shortest_path_to_token(arg_token, sentence[self.dependency_type], sentence['tokens'], 0, 0, 0, trigger_token, self.reduce_deps)
		trigger_dep_pattern_features_list = self.get_dep_pattern_features_list(self.trigger_dep_pattern_features, matching_trigger_dep_patterns)
		#trigger_dep_pattern_features_list = []
		dep_path_length = [gen_utils.get_dep_path_length(matching_trigger_dep_patterns[0])]
		# 8/23 added wn_type:
		wn_type = gen_utils.get_wn_type(arg_token['word'])
		wn_type_features_list = self.get_dep_pattern_features_list(self.wn_types_features, [wn_type])
		wn_type_features_list = []
		wn_types = gen_utils.get_wn_types(arg_token['word'])
		wn_types_features_list = self.get_dep_pattern_features_list(self.wn_types_features, wn_types)
		abs_distance = [self.get_abs_distance(arg_token, trigger_token)]
		relative_distance = [self.get_rel_distance(arg_token, trigger_token)]
		if arg_token == trigger_token:
			is_trigger = [0,1]
		else:
			is_trigger = [1,0]
		# 8/23 added local context:
		local_context_features = self.get_local_context_features(arg_token, sentence)
		#local_context_features = []
		#is_closest_ne
		distances = self.get_wn_distance_from_type_basket(arg_token)
		entity_type_features_list = self.get_dep_pattern_features_list(self.entity_types, [entity_type])
		return base_features + trigger_dep_pattern_features_list + wn_types_features_list + abs_distance + relative_distance + distances + dep_path_length + is_trigger + wn_type_features_list + local_context_features + entity_type_features_list
	def get_abs_distance(self, arg_token, trigger_token):
		return abs(self.get_rel_distance(arg_token, trigger_token))
	def get_rel_distance(self, arg_token, trigger_token):
		return arg_token['index'] - trigger_token['index']
	def get_wn_distance_from_type_basket(self, token):
		word = token['word']
		if word not in self.w2v_model:
			return map(lambda x: 1, self.arg_wordlist)
		#words_to_check = ['person', 'name', 'place', 'city', 'building', 'team', 'organization', 'time', 'clock', 'date', 'crime', 'punishment', 'weapon', 'money', 'dollars', 'beneficiary', 'recipient', 'giver', 'attacker', 'instrument', 'location']
		distances = []
		for word_to_check in self.arg_wordlist:
			if word_to_check not in self.w2v_model:
				distance = 1
			else:
				distance = gen_utils.cosine_distance(self.w2v_model[word], self.w2v_model[word_to_check])
			distances.append(distance)
		return distances
	def get_local_context_features(self, arg_token, sentence):
		if arg_token['index'] > 1:
			prev_token = self.get_token_by_index(arg_token['index'] - 1, sentence['tokens'])
			prev_ner_type = self.get_dep_pattern_features_list(self.ner_types, [prev_token['ner']])
			prev_word_match = self.get_dep_pattern_features_list(self.prev_word_match_list, [prev_token['word'].lower()])
		else:
			prev_ner_type = self.get_dep_pattern_features_list(self.ner_types, ['NONE'])
			prev_word_match = self.get_dep_pattern_features_list(self.prev_word_match_list, ['NONE'])
		if arg_token['index'] < len(sentence['tokens']):
			next_token = self.get_token_by_index(arg_token['index'] + 1, sentence['tokens'])
			next_ner_type = self.get_dep_pattern_features_list(self.ner_types, [next_token['ner']])
			next_word_match = self.get_dep_pattern_features_list(self.next_word_match_list, [next_token['word'].lower()])
		else:
			next_ner_type = self.get_dep_pattern_features_list(self.ner_types, ['NONE'])
			next_word_match = self.get_dep_pattern_features_list(self.next_word_match_list, ['NONE'])
		return prev_ner_type + next_ner_type + prev_word_match + next_word_match

			


