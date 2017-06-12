class NeighborDistance(object):
    """
    Simple data structure to store a near neighbor.
    @ivar doc_id: int - document id
    @ivar distance: float - distance
    """

    def __init__(self, doc_id, dist):
        self.doc_id = doc_id
        self.distance = dist
