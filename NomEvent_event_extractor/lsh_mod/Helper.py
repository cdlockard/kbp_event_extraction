import random


def create_projection_vectors(D, m):
    """
    Creates a set of projection vectors in the given dimension
    @param D: int - the dimension of the data
    @param m: int - the number of vectors to create
    """
    random.seed(4)
    return [[random.gauss(0.0, 1.0) for _ in xrange(D)] for _ in xrange(m)]
