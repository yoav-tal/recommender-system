
from sklearn.metrics.pairwise import cosine_similarity


class SimFunc:

    def __init__(self, k):
        self.name = None
        self.func = None
        self.k = k

    def __call__(self, embeddings):
        raise NotImplementedError("call function for embeddings data must be implemented by "
                                  "inherited SimFunc object")


class CosineSimilarity(SimFunc):

    def __init__(self, k):
        super().__init__(k)
        self.name = "cosine similarity"
        self.func = cosine_similarity

    def __call__(self, embeddings, *args, **kwargs):
        if embeddings.shape[0] == self.k:
            embeddings = embeddings.T
        if "subject_vectors" in kwargs.keys():
            subject_vectors = kwargs["subject_vectors"]
            subject_vectors  = subject_vectors.T if subject_vectors.shape[0] == self.k else \
                subject_vectors
            return self.func(subject_vectors, embeddings)

        return self.func(embeddings)


available_metrics = [CosineSimilarity]
