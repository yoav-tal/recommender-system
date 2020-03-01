import numpy as np


class ItemSimilarityModel:
    # The Model class abstracts the idea that a model is the combination of an algorithm that
    # learns low-rank embeddings, a dimension k of the embeddings and a similarity metric. The
    # model controls the calls to the algorithm’s fit() functions and computes the similarity
    # matrix from the embeddings.
    #
    # An instance is therefore initiated with a RecSysAlgo object, an integer k and a SimFunc
    # object. The Model class hosts a get_similarities function and a get_embeddings function
    # (both take data as argument).

    def __init__(self, algo, k, SimFunc):
        self.algo = algo
        self.k = k
        self.sim_func = SimFunc(k)
        self.embeddings = None
        self.embeddings_of = None
        self.similarities = None
        self.similarities_of = None

    def GetEmbeddings(self, data):
            if self.embeddings_of is not None and data.shape == self.embeddings_of.shape and \
                    (data.values == self.embeddings_of.values).all():
                return self.embeddings
            else:
                print("fitting algorithm %s with k=%g" % (self.algo.name, self.k))
                self.algo.fit(data, self.k)
                self.embeddings = self.algo.embeddings
                self.embeddings_of = data
                return self.embeddings

    def GetSimilarities(self, data):
        if self.similarities_of is not None and data.shape == self.similarities_of.shape and \
                (data.values == self.similarities_of.values).all():
            return self.similarities
        else:
            print("calculating similarities by algorithm %s with k=%g and similarity function "
                  "%s" % (self.algo.name, self.k, self.sim_func.name))
            self.similarities = self.sim_func(self.GetEmbeddings(data))
            self.similarities_of = data
            return self.similarities


class PopularityModel:
    def __init__(self):
        self.similarities = None

    def GetEmbeddings(self, data):
        raise NotImplementedError("Popularity Model is not based on embedding")

    def GetSimilarities(self, data):
        popularity_ranks = self.get_popularity_ranks(data)
        n_items = len(popularity_ranks)
        self.similarities = np.repeat([popularity_ranks], n_items, axis=0)
        return self.similarities

    def get_popularity_ranks(self, data):
        # get a list in which in the position of the least popular we have 0, the next least
        # popular is 1 and so on, the most popular is N (N the number of items)
        unique, counts = np.unique(data.template_index.values, return_counts=True)
        temp = np.argsort(counts)
        ranks = np.empty_like(temp)
        ranks[temp] = np.arange(len(temp))

        return ranks


if __name__ == "__main__":
    from algorithms.RecSysAlgo import SparseSVDAlgo
    from similarity_functions.SimFunc import CosineSimilarity
    import pandas as pd
    alg = SparseSVDAlgo
    simfunc = CosineSimilarity
    data = pd.read_csv("../data/RecSys data/min_previews 6.csv")

    algo = alg()
    k = 10
    model = Model(algo, k, simfunc)

    print(model.GetEmbeddings(data).shape)
    print(model.GetSimilarities(data).shape)
    print(model.GetSimilarities(data).shape)
