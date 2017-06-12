import math


def distance(doc1, doc2):
    dist = 0.0
    for word_id, count in doc1.iteritems():
        if doc2.has_key(word_id):
            dist += math.pow(doc2.get(word_id) - count, 2)
        else:
            dist += math.pow(count, 2)
    for word_id, count in doc2.iteritems():
        if not doc1.has_key(word_id):
            dist += math.pow(count, 2)
    return math.sqrt(dist)
